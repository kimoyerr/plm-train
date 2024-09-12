"""
This script trains a model using FSDP. It pulls inspiration from
- https://github.com/AnswerDotAI/fsdp_qlora/blob/main/train.py which in turn is inspired from:
    - llama-recipes (https://github.com/facebookresearch/llama-recipes/blob/main/src/llama_recipes/finetuning.py)
    - PyTorch FSDP docs (https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
    - bitsandbytes (https://github.com/TimDettmers/bitsandbytes)

For information on the different arguments, run `python train.py --help`
"""

# Imports

# General
import torch, os, gc, time, safetensors, copy, math, types
import functools
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import get_linear_schedule_with_warmup
import bitsandbytes as bnb
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import nullcontext
from safetensors.torch import save_file
from tqdm.auto import tqdm
from typing import List, Dict
import pandas as pd

# Argument parsing
from fastcore.script import call_parse, bool_arg, Param

# Torch + distributed training
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# FSDP
from torch.distributed.fsdp import MixedPrecision, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
from torch.distributed.fsdp.api import BackwardPrefetch, CPUOffload, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    offload_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.profiler import profile, record_function, ProfilerActivity

# Model loading
from bitsandbytes.nn import Linear4bit, Params4bit
from accelerate import init_empty_weights
from accelerate.utils import set_seed
from peft import get_peft_model, LoraConfig, TaskType
from peft.utils.other import fsdp_auto_wrap_policy
from transformers.utils import hub, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoConfig
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.models.esm.modeling_esm import EsmAttention, EsmSelfAttention
from fastcore.parallel import parallel

try:
    from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig
except ImportError:
    HQQLinear = None
    pass

# PEFT
from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder

# For different model types, we'll want to import the right class for the
# check_fn in activation checkpointing (LlamaDecoderLayer for llama models for example)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LLAMA_ATTENTION_CLASSES, LlamaMLP

# Contextual positional encodings
from cope_utils import CoPE, EsmSelfAttention_CoPE

# Import the confit eval function
from confit.train import evaluate

# To get rid of tokenizers warnings for now
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# For logging things during training
try:
    import wandb
except ImportError:
    pass

# Write logs from rank 0 only
class Logger:
    def __init__(self, args, log_to="stdout", project_name="fsdp_qlora", entity=None, group=None, name=None, rank=0):
        # self.log_every_n_steps = log_every_n_steps TODO: add this back as an option
        self.log_to = log_to
        if self.log_to == "wandb" and rank == 0:
            import wandb
            
            wandb.init(project=project_name, entity=entity, group=group, name=name, config=args)

    def log(self, d:Dict, rank:int):
        if rank != 0: 
            return
        if self.log_to == "tqdm":
            for k, v in d.items():
                tqdm.write(f'{k}: {v}')
        elif self.log_to == "wandb":
            wandb.log(d)
        elif self.log_to == "stdout":
            for k, v in d.items():
                print(k)
                print(f'{k}: {v}')

    def finish(self, rank=0):
        if self.log_to == "wandb" and rank == 0: wandb.finish()


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")


def update_progress_bar(progress_bar:tqdm, epoch:int, log_loss:float, log_lr:float, rank:int):
    """Updates the progress bar with the current epoch, loss, and learning rate"""
    if rank == 0:
        if log_lr >= 0:
            progress_bar.set_description(f"Epoch {epoch}, Loss {log_loss:.3f}, LR {log_lr:.2e}", refresh=True)
        else:
            progress_bar.set_description(f"Epoch {epoch}, Loss {log_loss:.3f}", refresh=True)


def setup_quantized_meta_for_peft(model:nn.Module):
    """Replaces `quant_state.to` with a dummy function to prevent PEFT from moving `quant_state` to meta device"""
    def temp_to_method(self, *args, **kwargs):
        return self
    for param in model.parameters():
        if isinstance(param, Params4bit):
            param.quant_state._orig_to = param.quant_state.to
            param.quant_state.to = types.MethodType(temp_to_method, param.quant_state)


def setup_quantized_peft_meta_for_training(model:nn.Module):
    """Replaces dummy `quant_state.to` method with the original function to allow training to continue"""
    for param in model.parameters():
        if isinstance(param, Params4bit) and hasattr(param.quant_state, '_orig_to'):
            param.quant_state.to = param.quant_state._orig_to
            param.quant_state._orig_to = None


# Dataloader for Confit
def get_confit_dataloader(args:Dict):

    from transformers import EsmTokenizer
    from confit.data_utils import Mutation_Set

    dataset_name = args["protein_dataset"]

    train_csv = pd.read_csv(args["protein_trainset_path"])
    test_csv = pd.read_csv(args["protein_testset_path"])
    val_csv = pd.read_csv(args["protein_valset_path"])
    
    model_seed = 1
    tokenizer = EsmTokenizer.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_{model_seed}')
    trainset = Mutation_Set(data=train_csv, fname=dataset_name, tokenizer=tokenizer)
    valset = Mutation_Set(data=val_csv, fname=dataset_name,  tokenizer=tokenizer)
    testset = Mutation_Set(data=test_csv, fname=dataset_name,  tokenizer=tokenizer)
    print(len(trainset), len(valset), len(testset))

    # For distributed training, use DistributedSampler
    sampler = DistributedSampler(trainset, seed=args["seed"])

    # No collate function since trainset already has one
    dataloader = DataLoader(trainset, batch_size=args["batch_size"], collate_fn=trainset.collate_fn, sampler=sampler)
    valloader = DataLoader(valset, batch_size=args["batch_size"], collate_fn=valset.collate_fn)
    testloader = DataLoader(testset, batch_size=args["batch_size"], collate_fn=testset.collate_fn)

    return dataloader, valloader, testloader


# LR scheduler.
def _get_cosine_one_cycle_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, min_lr_fraction = 0.1,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    scale_term = (1 - min_lr_fraction)
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return (math.cos(math.pi * progress)+1) * 0.5 * scale_term + min_lr_fraction


def get_cosine_one_cycle_scheduler(optimizer:optim.Optimizer, num_warmup_steps:int, num_training_steps:int, min_lr_fraction:float=0.1):
    "A more general cosine scheduler with to control the minimum learning rate"
    lr_lambda = functools.partial(
        _get_cosine_one_cycle_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_fraction=min_lr_fraction
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def get_lr_scheduler(optimizer:optim.Optimizer, dataloader:DataLoader, gradient_accumulation_steps:int, args:Dict):
    """Returns linear, cosine, or constant learning rate scheduler"""
    num_training_steps = args['num_epochs'] * len(dataloader) // gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * 0.1)
    if args['lr_scheduler'] == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args['lr_scheduler'] == "cosine":
        lr_scheduler = get_cosine_one_cycle_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr_fraction=0.1)
    elif args['lr_scheduler'] == "constant":
        lr_scheduler = None
    else:
        raise NotImplementedError(f"{args['lr_scheduler']} LR scheduler not implemented yet")
    return lr_scheduler, num_training_steps


# Optimizer
def get_optimizer(model:nn.Module, args:Dict):
    """Returns an optimizer. We can add more options here if needed."""
    if args["optimizer"] == "adam":
        return optim.Adam(model.parameters(), lr=args['lr'])
    elif args["optimizer"] == "sgd":
        return optim.SGD(model.parameters(), lr=args['lr'])
    elif args["optimizer"] == "adadelta":
        return optim.Adadelta(model.parameters(), lr=args['lr'])
    elif args["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args['lr'], betas=(0.9,0.95),
                                 eps=1e-5, weight_decay=args['wd'])
    else:
        raise ValueError("Invalid optimizer")


# Wrap the model using LoRA policy from llama-recipes or custom policy:
# Copied from peft.utils.other and modified the exception when the transformer layer class could not be found to a warning
def fsdp_auto_wrap_policy_confit(model):

    # from accelerate import FullyShardedDataParallelPlugin
    from accelerate.utils.dataclasses import get_module_class_from_name
    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    default_transformer_cls_names_to_wrap = (
        ",".join(model._no_split_modules) if getattr(model, "_no_split_modules", None) is not None else ""
    )
    transformer_cls_names_to_wrap = os.environ.get(
        "FSDP_TRANSFORMER_CLS_TO_WRAP", default_transformer_cls_names_to_wrap
    ).split(",")
    transformer_cls_to_wrap = {PrefixEncoder, PromptEncoder, PromptEmbedding}
    for layer_class in transformer_cls_names_to_wrap:
        transformer_cls = get_module_class_from_name(model, layer_class)
        if transformer_cls is None:
            # Add a warning here instead of raising an exception
            print(f"Warning: Transformer layer class {layer_class} not found in model")
        else:
            transformer_cls_to_wrap.add(transformer_cls)

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_cls_to_wrap,
    )

    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_wrap_policy

# for tt in model.modules():
#     if "Attention" in type(tt).__name__:
#         print(type(tt))
#         print(type(tt).__name__)
#         print(getattr(tt, "weight", None))


# Custom LORA module.
class LORA(nn.Module):
    def __init__(self, base_layer, lora_rank, lora_alpha, lora_dropout):
        super().__init__()
        self.base_layer = base_layer
        dtype = getattr(base_layer, "compute_dtype", next(base_layer.parameters()).dtype)
        device = next(base_layer.parameters()).device
        lora_A = nn.Linear(base_layer.in_features, lora_rank, bias=False, device=device, dtype=dtype)
        lora_B = nn.Linear(lora_rank, base_layer.out_features, bias=False, device=device, dtype=dtype)
        lora_B.weight.data.zero_()

        self.lora_AB = nn.Sequential(lora_A, lora_B)

        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.scaling = self.lora_alpha / lora_rank

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        result = self.base_layer(x, *args, **kwargs)
        # As per Tim Dettmers, for 4bit, we need to defensively clone here.
        # The reason is that in some cases, an error can occur that backprop
        # does not work on a manipulated view. This issue may be solved with
        # newer PyTorch versions but this would need extensive testing to be
        # sure.
        result = result.clone()

        requires_conversion = not torch.is_autocast_enabled()
        if requires_conversion:
            expected_dtype = result.dtype
            x = x.to(next(iter(self.lora_AB)).weight.dtype)

        output = self.lora_AB(self.lora_dropout(x))
        if requires_conversion:
            output = output.to(expected_dtype)
        output = output * self.scaling

        result += output

        return result


# Function to replace a model's module
def replace_module(module, target_name, model_config):
    for child_name, child_module in module.named_children():
        if target_name in type(child_module).__name__:
            new_module = EsmSelfAttention_CoPE(model_config)
            setattr(module, child_name, new_module)
            # print(module)
        else:
            replace_module(child_module, target_name, model_config)


# Save checkpoint: Modified from https://github.com/pytorch/torchtune/blob/main/recipes/lora_finetune_distributed.py
def save_checkpoint(
    model,
    optimizer,
    epoch,
    total_epochs,
    rank,
    output_dir,
):
    """
    Checkpoint the state of the recipe. The constructed checkpoint state dict
    contains the following information:
    - Merged weights with key MODEL_KEY
    - Adapter weights with key ADAPTER_KEY
    - Relevant recipe state if training is not complete

    Checkpointer will save the merged weights, adapter weights and recipe state in
    different checkpoint files. To correctly resume from training, the adapter weights
    and recipe state must be provided along with the base model weights.
    """
    # final dict passed onto the checkpointer
    checkpoint_dict = {}

    intermediate_checkpoint = epoch + 1 < total_epochs
    # To prevent GPU memory from spiking during checkpoint save,
    # we consolidate the full model and optim state dicts on CPU for rank 0
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state_dict = model.state_dict()
        if intermediate_checkpoint:
            # opt_state_dict = FSDP.optim_state_dict(model, optimizer)
            opt_state_dict = None  # For now, lets turn this off since we don't need to use these optimizer state to resume training.
        else:
            opt_state_dict = None

    # Now that we have the model and opt state dict, create the actual checkpoint dict
    # to be sent to the checkpointer and ultimately written to file
    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("Saving full model weights: intermediate checkpoint")
        if epoch == -1:  # Before training
            epoch = "start"
        save_file(cpu_state_dict, os.path.join(output_dir, f"model_state_dict_epoch_{epoch}.safetensors"))
        print("Done", rank)


# Main function, run on each process
def fsdp_main(local_rank:int, world_size:int, args:Dict):
    print_func = tqdm.write if args["log_to"] == 'tqdm' else print

    # Setup and initialize the process group
    os.environ['MASTER_ADDR'] = args["master_addr"]
    os.environ['MASTER_PORT'] = args["master_port"]
    if 'SLURM_PROCID' in os.environ:
        # assumes same number of GPUs per node.
        rank = int(os.environ['SLURM_PROCID']) * torch.cuda.device_count() + local_rank
    else:
        rank = local_rank

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(local_rank)
    

    # Start logging
    if args["group"] is None:
        args["group"] = args["protein_dataset"]
    logger = Logger(args, log_to=args["log_to"], project_name=args["project_name"],
                    entity=args["entity"], group=args["group"], name=args["name"], rank=rank)

    # Timing stuff
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # model precision, qlora compute precison, and FSDP mixed precision policy.
    # The Linear4Bit quant_storage dtype should always match the FSDP param_dtype. The compute_dtype should match the AMP compute dtype.
    # MixedPrecision(param_dtype=fp32, reduce_dtype=fp32, buffer_dtype=fp32) uses `torch.amp.autocast` to control precision.
    # limited qlora testing shows that fp16 only works with autocast while bf16 trains with both pure and autocast modes.
    # TODO: test how often this holds for mp_fp16
    mp_policy = None
    load_param_skip_names = []
    if args["precision"] == "bf16":
        torch_dtype, compute_dtype = torch.bfloat16, torch.bfloat16
    elif args["precision"] == "fp32":
        torch_dtype, compute_dtype = torch.float32, torch.float16
    elif args["precision"] == "fp16_autocast":
        compute_dtype, torch_dtype = torch.float16, torch.float32
        mp_policy = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    elif args["precision"] == "bf16_autocast":
        compute_dtype, torch_dtype = torch.bfloat16, torch.float32
        mp_policy = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    elif args["precision"] == "bf16_buffers_autocast":
        compute_dtype, torch_dtype = torch.bfloat16, torch.bfloat16
        mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.float32)
        load_param_skip_names = ['inv_freq']
    else:
        raise ValueError("Invalid precision")


    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
    tokenizer.pad_token_id = tokenizer.eos_token_id # TODO check if it exists first

    # Set up dataloader
    if args["dataset"]!="confit":
        dataloader = get_dataloader(tokenizer, args)
    else:
        dataloader, valloader, testloader = get_confit_dataloader(args)



    # Create model
    cfg = None
    attn_impl = "sdpa" # torch 2.2 sdpa uses flash attn 2
    print("Creating model", rank)
    if args["train_type"] in ["full", "lora", "dora", "custom_lora"]:
        if (args["low_memory"] and rank == 0) or (not args["low_memory"]):
            if args["model_type"] == "masked":
                model = AutoModelForMaskedLM.from_pretrained(
                    args["model_name"],
                    use_cache=False,
                    torch_dtype=torch_dtype,
                    _attn_implementation="eager"

                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args["model_name"],
                    use_cache=False,
                    torch_dtype=torch_dtype,
                    _attn_implementation=attn_impl
                )
            dtype = torch_dtype if args["precision"] == "bf16" else None
            model.to(dtype=dtype, device="cpu" if args["low_memory"] else rank)
        else:
            cfg = AutoConfig.from_pretrained(args["model_name"])
            cfg.use_cache = False
            cfg._attn_implementation = attn_impl
            with init_empty_weights():
                if args["model_type"] == "masked":
                    model = AutoModelForMaskedLM.from_config(cfg, torch_dtype=torch_dtype)
                else:
                    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch_dtype)
            if args["precision"] == "bf16":
                model.to(torch_dtype)
    elif args["train_type"] in ["qlora", "custom_qlora", "hqq_lora"]: # Our custom loading
        cfg = AutoConfig.from_pretrained(args["model_name"])
        cfg.use_cache = False
        cfg._attn_implementation = attn_impl

        # load model on meta device without calling init and replace nn.Linear with Linear4bit
        with init_empty_weights():
            if args["model_type"] == "masked":
                model = AutoModelForMaskedLM.from_config(cfg)
            else:
                model = AutoModelForCausalLM.from_config(cfg)
            if args["train_type"] in ["hqq_lora"]:
                # TODO: Tune BaseQuantizeConfig.
                quant_config = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True,
                                                  quant_scale=True, offload_meta=True, view_as_float=True)
                model.model = replace_linear(model.model, HQQLinear, quant_config, device=rank,
                                             compute_dtype=compute_dtype, del_orig=True, initialize=False)
                HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)
            else:
                model.model = replace_linear(model.model, Linear4bit, compute_dtype=compute_dtype,
                                             quant_type='nf4', quant_storage=torch_dtype)
        model.is_loaded_in_4bit = True

        # Grab the safetensors files that hold the weights
        try:
            idx = hub.cached_file(args["model_name"], SAFE_WEIGHTS_INDEX_NAME)
            files, _ = hub.get_checkpoint_shard_files(args["model_name"], idx)
        except OSError:
            try:
                # This means the model doesn't have a model.safetensors.index.json because it is not sharded
                files = []
                files.append(hub.cached_file(args["model_name"], SAFE_WEIGHTS_NAME))
            except OSError as e:
                # This means the model probably doesn't have a safetensors file
                raise e

        # Load in the weights, using our custom load_and_quantize method which quantizes Params4bit on the fly
        # and then places each layer on CPU or meta if using low_memory to minimize GPU memory usage
        def load_and_quantize_parallel(name_param, model, **kwargs):
            name, param = name_param
            load_and_quantize(model, name, param, **kwargs)

        print("Loading model", rank)
        param_count = sum((p.numel() for n,p in model.named_parameters()))
        if rank == 0 and args['verbose']:
            print_func(f"Total model params: {param_count}")
        start = time.time()
        for filename in files:
            weights = safetensors.torch.load_file(filename)
            quant_method = "hqq" if args["train_type"] in ["hqq_lora"] else "bnb"
            devprops = torch.cuda.get_device_properties(torch.cuda.current_device())
            left = int(os.cpu_count()/torch.cuda.device_count())
            right = int(8 * (devprops.total_memory/1e9/40) * (70/(param_count/1e9)))
            n_workers = min(left, right)
            if rank == 0 and args['verbose']:
                print_func(f"Using n_workers: {n_workers} for loading")
            parallel(load_and_quantize_parallel, iter(weights.items()), n_workers=n_workers, threadpool=True,
                     model=model, dtype=torch_dtype, device=local_rank, skip_names=load_param_skip_names,
                     is_meta_rank=(args["low_memory"] and rank!=0), verbose=args["verbose"], quant_method=quant_method)
        if rank == 0 and args["verbose"]:
            print(f"Loaded model weights in {time.time()-start:.3f} seconds")

    print("Model created", rank, f"{torch.cuda.memory_allocated(local_rank)/1e9:.3f} GB")


    # PEFT setup (LoRA and QLoRA)
    if args["train_type"] in ["lora", "qlora", "dora"]:
        if args["train_type"] == "dora":
            use_dora = 1
        else:
            use_dora = 0
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False,
            r=args["lora_rank"],
            lora_alpha=args["lora_alpha"],
            lora_dropout=args["lora_dropout"],
            target_modules=args["lora_target_modules"],
            use_dora=use_dora,
        )
        # PEFT will move quant_state to meta device, so this method prevents that
        # from happening by replacing quant_state.to with a dummy function
        if rank!=0 and args["low_memory"]:
            setup_quantized_meta_for_peft(model)

        # Replace self-attention with self-attention with CoPE
        # TODO: To make this work on smaller GPU memories, we have to optimize the memory usage. It works currently only on 320GB GPU memory (4 X A100)
        # replace_module(model, "EsmSelfAttention", model.config)

        # Modify model 
        model = get_peft_model(model, peft_config)

        if rank==0:
            model.print_trainable_parameters()
        elif args['low_memory']:
            # And then setup_quantized_peft_meta_for_training sets quant_state.to back to normal
            setup_quantized_peft_meta_for_training(model)
    elif args["train_type"] in ["custom_qlora", "custom_lora", "hqq_lora"]:
        # Create LORA layers.
        for name, _ in model.named_modules():
            module_key, _, value_key = name.rpartition('.')
            if value_key in args['lora_target_modules']:
                m = model.get_submodule(name)
                qlora_layer = LORA(m, args["lora_rank"], args["lora_alpha"], args["lora_dropout"])
                parent_module = model.get_submodule(module_key)
                setattr(parent_module, value_key, qlora_layer)
        for n,p in model.named_parameters():
            if any([lora_name in n for lora_name in ['lora_AB', 'lora_A', 'lora_B']]):
                p.requires_grad = True
                if args['verbose']:
                    print("Trainable LORA layer", n)
            else:
                p.requires_grad = False

        print("LoRA layers added", rank, f"{torch.cuda.memory_allocated(local_rank)/1e9:.3f} GB")

    logger.log({"memory_after_model_creation": torch.cuda.memory_allocated(local_rank)}, rank)


    # Wrap model with llama-recipies or custom LoRA policy
    if args["dataset"] == "confit":
        my_auto_wrap_policy = fsdp_auto_wrap_policy_confit(model)

    print("Wrapping model w/ FSDP", rank)
    if args["sharding_strategy"] == "full_shard":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif args["sharding_strategy"] == "shard_grad_op":
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif args["sharding_strategy"] == "ddp":
        sharding_strategy = ShardingStrategy.NO_SHARD
    elif args["sharding_strategy"] == "hybrid_full_shard":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif args["sharding_strategy"] == "hybrid_shard_grad_op":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
    else:
        raise ValueError("Invalid FSDP sharding strategy")

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=my_auto_wrap_policy,
        # backward_prefetch=None, #BackwardPrefetch.BACKWARD_PRE
        use_orig_params=False,
        cpu_offload=CPUOffload(offload_params=True) if args["use_cpu_offload"] else None,
        limit_all_gathers=True,  # See https://github.com/pytorch/pytorch/issues/91165
        device_id=torch.cuda.current_device(),
        sync_module_states=args["low_memory"],
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if (rank!=0 and args["low_memory"]) else None,  # TODO note about meta device and why we need this
        mixed_precision=mp_policy,
    )
    print("Wrapped model", rank, f"{torch.cuda.memory_allocated(local_rank)/1e9:.3f} GB")
    logger.log({"memory_after_model_wrap": torch.cuda.memory_allocated(local_rank)}, rank)

    # Synchronize at the start
    dist.barrier()

    # TODO: Test with torch.compile
    # model = torch.compile(model)

    # Apply activation checkpointing
    if args["use_gradient_checkpointing"]:
        if args['reentrant_checkpointing']:
            model.enable_input_require_grads()
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.REENTRANT if args['reentrant_checkpointing'] else CheckpointImpl.NO_REENTRANT,
        )

        check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
        print("Applying activation checkpointing", rank)
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )

    if args["use_activation_cpu_offload"]:
        print("Applying activation offloading", rank)
        model = offload_wrapper(model)

    if rank == 0 and args['verbose']:
        print("Config:")
        print(cfg)
        print("Model:")
        print(model)
        print("Starting training")


    # Create the optimizer
    optimizer = get_optimizer(model, args)


    # LR scheduler.
    gradient_accumulation_steps = max(1, args['gradient_accumulation_steps'])
    lr_scheduler, num_training_steps = get_lr_scheduler(optimizer, dataloader, gradient_accumulation_steps, args)

    # Sanity check: see what parameters the optimizer has and which require grad:
    if rank == 0 and args['verbose']:
        print("Optimizer params:")
        for group in optimizer.param_groups:
            for param in group['params']:
                print(f"Shape: {param.shape}, Requires Grad: {param.requires_grad}, Dtype: {param.dtype}")


    # Autocast for mixed precision with fp16/bf16 compute types with fp32 params
    if args["precision"] in ["fp16_autocast", "bf16_autocast", "bf16_buffers_autocast"]:
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=compute_dtype)
    else:
        autocast = nullcontext()
    scaler = ShardedGradScaler() if args["precision"] == "fp16_autocast" else None
    scale_grads = scaler is not None


    if rank == 0:
        print("Total Training Steps:", num_training_steps)
    progress_bar = tqdm(range(num_training_steps), disable=rank != 0)
    init_start_event.record()
    log_loss, log_lr = 0.0, -1
    
    # Set context manager for memory tracking
    if args["torch_profile_memory"]:
        memory_profile_context = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=True,
                        on_trace_ready=trace_handler,
                        )
    else:
        memory_profile_context = nullcontext()
    # Reset peak memory to track that
    torch.cuda.reset_peak_memory_stats(local_rank)

    # Run epochs with memory profiler
    with memory_profile_context as prof:
        total_batches = 0
        for epoch in range(args['num_epochs']):
            
            update_progress_bar(progress_bar, epoch, log_loss, log_lr, rank)
            model.train()
            ddp_loss = torch.zeros(2).to(local_rank)

            total_loss = torch.zeros(1).to(local_rank)

            # Save model before starting
            # TODO: Also evaluate the model before adding finetuning layers like LoRA and others
            # Evaluate model before first iteration
            if args["dataset"] == "confit" and epoch == 0:
                with torch.no_grad():
                    model.eval()
                    print(f"Validating")
                    sr = evaluate(model, valloader, tokenizer, accelerator="fsdp")
                    print(f'========epoch{epoch}; val spearman correlation :{sr}=================')
                    logger.log({"val_spearman": sr}, rank)
                    # Test
                    print(f"Testing")
                    sr = evaluate(model, testloader, tokenizer, accelerator="fsdp")
                    print(f'========epoch{epoch}; test spearman correlation :{sr}=================')
                    logger.log({"test_spearman": sr}, rank)

                    # save model    
                    if 1:
                        save_checkpoint(    
                            model,
                            optimizer,
                            epoch = epoch,
                            total_epochs = args['num_epochs'],
                            rank = rank,
                            output_dir = args["output_dir"]
                        )
                        # dist.barrier()  
                    model.train()
            

            
            # Run through all batches
            for batch_idx, batch in enumerate(dataloader):
                total_batches += 1
                # print(epoch, batch_idx)
                if args["torch_profile_memory"]:
                    prof.step()
                accumulate_grads = (batch_idx+1) % gradient_accumulation_steps == 0

                # Prevent gradient syncing until update step if using no_sync option.
                # Documentation states this should only be used on the root FSDP instance
                # We assume this is a one-node setup
                if args['no_sync'] and not accumulate_grads:
                    sync_context = model.no_sync()
                else:
                    sync_context = nullcontext()

                # Start logging memory (first iter) if requested
                if batch_idx==0 and rank == 0 and epoch == 0 and args['profile_memory']:
                    torch.cuda.memory._record_memory_history()

                # Log memory usage
                if batch_idx == 4 and epoch == 0:
                    logger.log({"memory_before_forward": torch.cuda.memory_allocated(local_rank)/1e9}, rank)

                # Forward pass
                with sync_context:
                    # Forward pass specific to ConFit finetuning approach
                    if args["dataset"] == "confit":
                        # TODO: Can we move this outside the context?
                        from confit.stat_utils import compute_score, BT_loss, BT_loss_disco, BT_loss_disco_gather

                        with autocast:
                            with record_function("## forward ##"):
                                # compute the score based on the masked amino acids
                                seq, mask = batch[0], batch[1]
                                wt, wt_mask = batch[2], batch[3]
                                pos = batch[4]
                                golden_score = batch[5]
                                score, logits, out, log_probs = compute_score(model, seq, mask, wt, pos, tokenizer)
                                score = score.to(local_rank)
                                golden_score = golden_score.to(local_rank)

                                # TODO: Add regularization wrt original model to prevent catastrophic forgetting
                                # out_reg = model_reg(wt, wt_mask)
                                # logits_reg = out_reg.logits
                                # l_reg = KLloss(logits, logits_reg, seq, mask)
                                # loss = l_BT + lambda_reg*l_reg

                                # BT loss
                                disco = 1
                                if disco:
                                    loss, local_scores, all_scores, all_golden_scores = BT_loss_disco_gather(local_rank, score, golden_score)
                                else:
                                    loss = BT_loss(score, golden_score)

                    else:
                        with autocast:
                            output = model(
                                batch['input_ids'].to(local_rank),
                                labels=batch['labels'].to(local_rank),
                                attention_mask=None,
                            )
                            loss = output.loss

                    # Scale loss for gradient accumulation
                    total_loss += loss.item()
                    loss = loss / gradient_accumulation_steps

                    # Log memory usage
                    if batch_idx == 4 and epoch == 0:
                        logger.log({"memory_after_forward": torch.cuda.memory_allocated(local_rank)/1e9}, rank)

                    # Backward pass
                    if scale_grads:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                        # TODO: Double-check the backward pass over the gradients is correct
                        # if args["dataset"] == "confit" and disco:
                        #     scores_grad = all_scores.grad
                        #     torch.distributed.all_reduce(scores_grad, op=torch.distributed.ReduceOp.AVG)
                        #     local_scores.backward(scores_grad[batch[0].shape[0]*local_rank:batch[0].shape[0]*(local_rank+1)])


                # Record loss
                bs = batch[0].shape[0]
                ddp_loss[0] += loss.item() * bs * gradient_accumulation_steps
                ddp_loss[1] += bs

                # Step the optimizer (w/ gradient accumulation)
                if accumulate_grads:
                    if args['apply_gradient_clipping'] and (args['grad_norm'] is not None):
                        model.clip_grad_norm_(args['grad_norm'], norm_type=2.0)
                    if scale_grads:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # optimizer.zero_grad(set_to_none=True)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # model.zero_grad(set_to_none=True)
                    # Set other grads to zero
                    if args["dataset"] == "confit":
                        if disco:
                            all_scores.grad = None
                            local_scores.grad = None
                            scores_grad = None
                            all_golden_scores.grad = None
                            out.grad = None
                            log_probs.grad = None
                        golden_score.grad = None
                        score.grad = None
                        logits.grad = None
                        loss.grad = None

                    # avoid overhead when lr is constant.
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    progress_bar.update(1)

                # Log memory usage after backwards
                if batch_idx == 4 and epoch == 0:
                    logger.log({"memory_after_backward": torch.cuda.memory_allocated(local_rank)/1e9}, rank)

                # Delete the output so more memory frees up before the next forward pass
                output = None
                loss = None
                logits = None
                score = None
                golden_score = None
                if disco:
                    local_scores = None
                    all_scores = None
                    all_golden_scores = None
                    out = None
                    log_probs = None

                # Stop logging memory (first iter)
                if batch_idx == 0 and rank == 0 and epoch == 0 and args['profile_memory']:
                    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
                    with open("memory_snapshot.pickle", 'rb') as f:
                        loaded_data = pickle.load(f)
                    memory_df = pd.read_pickle("memory_snapshot.pickle")

                    torch.cuda.memory._record_memory_history(enabled=None) # Stop recording

                # Log loss every gradient update steps
                if accumulate_grads:
                    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
                    # dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                    if rank == 0:
                        log_loss = ddp_loss[0] / ddp_loss[1]
                        print(ddp_loss)
                        print(total_loss)
                        if lr_scheduler is not None:
                            log_lr = lr_scheduler.get_last_lr()[0]
                        else:
                            log_lr = args["lr"]
                        update_progress_bar(progress_bar, epoch, log_loss, log_lr, rank)
                        if args["log_to"] == 'wandb':
                            logger.log({"loss": log_loss, "lr": log_lr}, rank)
                    ddp_loss = torch.zeros(2).to(local_rank)

                    torch.cuda.empty_cache()
                else:
                    torch.cuda.empty_cache()
            
            #  End running through batches for this epoch
            
            # Evaluate model after certain number of epochs
            if epoch % args["eval_interval"] == 0:
                with torch.no_grad():
                    model.eval()
                    print(f"Validating")
                    sr = evaluate(model, valloader, tokenizer, accelerator="fsdp")
                    print(f'========epoch{epoch}; val spearman correlation :{sr}=================')
                    logger.log({"val_spearman": sr}, rank)
                    # Test
                    print(f"Testing")
                    sr = evaluate(model, testloader, tokenizer, accelerator="fsdp")
                    print(f'========epoch{epoch}; test spearman correlation :{sr}=================')
                    logger.log({"test_spearman": sr}, rank)

                    # save model    
                    if 1:
                        save_checkpoint(    
                            model,
                            optimizer,
                            epoch = epoch,
                            total_epochs = args['num_epochs'],
                            rank = rank,
                            output_dir = args["output_dir"]
                        )
                        # dist.barrier()  
                    model.train()
            

            # Print + log peak memory usage for the whole first step of training
            if epoch == 0 and rank == 0:
                peak_memory = torch.cuda.max_memory_allocated(local_rank)
                if args["verbose"]:
                    print_func(f"Peak memory usage (training): {peak_memory/1e9:.2f}GB", rank)
                if args["log_to"] == 'wandb':
                    logger.log({"memory_peak": peak_memory}, rank)

            if rank == 0:
                print(total_loss)
            elif rank == 1:
                print(total_loss)


    # Synchronize at the end and record time
    # if args["log_memory"]:
    #     prof.export_memory_timeline(f"confit_memory.html", device="cuda:"+str(local_rank))
        # prof.export_memory_timeline(f"confit_memory.html", device="cuda:1")
    init_end_event.record()
    dist.barrier()
    torch.cuda.synchronize()
    
    if rank == 0:
        print("Finished training", rank)

    # Print time and model
    time_taken = init_start_event.elapsed_time(init_end_event) / 1000
    dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        print(f"CUDA event elapsed time: {time_taken} sec")
        logger.log({"time_taken": time_taken}, rank)

    # End logging
    logger.finish(rank=rank)

    # Save model - ref: https://github.com/pytorch/pytorch/issues/98823
    # HQQLinear custom state_dict() method causes issues when saving.
    # Model is saved fine when `state_dict()` method is removed.
    # Non param/buffer types are not saved with FSDP.
    # It might be better to just save the trained lora layers.
    # summon_full_params on lora layers and save.
    if args["save_model"]:
        if rank == 0:
            os.makedirs(args["output_dir"], exist_ok=True)
        dist.barrier()
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        if args["train_type"] in ["custom_lora", "custom_qlora", "hqq_lora"]:
            cpu_state_dict = {}
            trainable_modules = [(n,m) for n,m in model.named_modules() if n.endswith('lora_AB')]
            for prefix, module in trainable_modules:
                prefix = (prefix.replace("_fsdp_wrapped_module.", "")
                                .replace("_checkpoint_wrapped_module.", "")
                                .replace("_offload_wrapped_module.", ""))
                with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, save_policy):
                    cpu_state_dict = {**cpu_state_dict, **{f"{prefix}.{k}":v for k,v in module.state_dict().items()}}
                dist.barrier()
                torch.cuda.synchronize()
            if rank==0:
                print_func("Saving trained LoRA weights.")
                save_file(cpu_state_dict, os.path.join(args["output_dir"], "model_state_dict.safetensors"))
                print_func("Done", rank)
        else:
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state_dict = model.state_dict()
                if rank==0:
                    print_func("Saving full model weights.")
                    save_file(cpu_state_dict, os.path.join(args["output_dir"], "model_state_dict.safetensors"))
                    print_func("Done", rank)

    dist.barrier() # Stop other processes ending while model saving - probably not needed?

    # Clean up
    dist.destroy_process_group()


# Entry point, using fastcore's call_parse to parse args from command line and then calling fsdp_main
@call_parse()
def main(
    world_size: int = -1, # Number of GPUs to use. -1 = all available GPUs.
    train_type: Param("", choices=["full", "lora", "dora", "qlora", "custom_qlora", "custom_lora", "hqq_lora"]) = "qlora", # "full", "lora", "qlora", or "custom_qlora"
    batch_size: int = 1, # Batch size per GPU. Effective BS = batch_size * world_size * gradient_accumulation_steps
    context_length: int = 512, # Max length of input sequence (in tokens)
    gradient_accumulation_steps: int = 1, # How many steps to accumulate gradients over (increases effective batch size)
    num_epochs: int = 100, # How many epochs of training to do
    dataset: Param("", choices=["alpaca", "alpaca_sample", "dummy", "guanaco", "sql", "confit"]) = "alpaca_sample", # alpaca, alpaca_sample (for a 128-sample test) or "dummy" for 16 long dummy samples
    protein_dataset: str = None, # Protein dataset to use. If None, uses the default dataset for the model
    sharding_strategy: Param("", choices=["full_shard", "shard_grad_op", "ddp", "hybrid_full_shard", "hybrid_shard_grad_op"]) = "full_shard", # Sharding strategy for FSDP
    use_gradient_checkpointing: bool_arg = True, # Use FSDP's activation checkpointing
    reentrant_checkpointing: bool_arg = False, # Use re-entrant autograd activation checkpointing. Setting to True can use less GPU memory with BNB QLoRA
    use_cpu_offload: bool_arg = True, # Use FSDP's CPU offloading
    use_activation_cpu_offload: bool_arg = False, # Use FSDP's activation CPU offloading
    low_memory: bool_arg = True, # Load one copy of the model into CPU memory before sharding with FSDP. For QLoRA, quantizes each layer individually on GPU before placing on CPU.
    no_sync: bool_arg = False, # Prevent gradient sync until update step. Likely uses more memory. Required for `use_cpu_offload` and `gradient_accumulation_steps > 1`
    precision: Param("", choices=["fp32", "bf16", "fp16_autocast", "bf16_autocast", "bf16_buffers_autocast"]) = "bf16", # Training precision. autocast precisions use mixed precision
    model_name: str = "meta-llama/Llama-2-7b-hf", # Which model to train - e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_type: str = "causal", # Model type to load. e.g. "causal" or "masked"
    save_model: bool_arg = False, # Save the resulting model
    output_dir: str = "output", # Output directory to save the final model to
    lora_rank: int = 64, # LoRA rank for lora/qlora
    lora_alpha: int = 16, # LoRA alpha for lora/qlora
    lora_dropout: float = 0.1, # LoRA dropout for lora/qlora
    lora_target_modules: str = "all", # If 'default', uses peft defaults. Use 'all' for our best guess for Llama models
    verbose: bool_arg = False, # Whether to print extra info for debugging
    lr: float = 1e-5, # Learning rate
    apply_gradient_clipping: bool_arg = False, # Apply gradient norm clipping
    grad_norm: float = 0.3, # Gradient norm clipping
    wd: float = 0.1, # Weight decay
    profile_memory: bool_arg = False, # Profile memory usage for the first few batches. Keep false for training. May increase memory usage.
    optimizer: Param("", choices=["adamw", "adam", "sgd", "adadelta"]) = "adamw", # Optimizer
    lr_scheduler: Param("", choices=["constant", "linear", "cosine"]) = "constant", # Learning Rate Scheduler. linear and cosine warm up for 10% of training steps.
    log_to: Param("", choices=["tqdm", "wandb", "stdout"]) = "tqdm", # Where to log output
    master_addr: str = "localhost", # For distributed training
    master_port: str = "12355", # For distributed training, must be the same for all processes
    seed: int = 42, # Random seed
    eval_interval: int = 10, # After how many epochs should we evaluate the model
    model_save_interval: int = 10, # After how many epochs should we save the model
    project_name: str = "fsdp_qlora", # For wandb logging
    name: str = None, # For wandb logging
    group: str = None, # For wandb logging
    entity: str = None, # For wandb logging
    torch_profile_memory: bool_arg = False, # Profile memory using pytorch
    protein_trainset_path: str = None, # Training file
    protein_valset_path: str = None, # Validation file
    protein_testset_path: str = None, # Test file
):

    # Set world size
    if world_size == -1:
        world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")

    # Get all args which will be passed to fsdp_main
    args = dict(locals())
    set_seed(args['seed'])
    if args['verbose']: print(args)

    # If lora_target_modules is 'all', set sensible defaults for llama + mistral type modules
    # See peft.utils.constants -> TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING for the current defaults
    if lora_target_modules == "all":
        args["lora_target_modules"] = ["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
    elif lora_target_modules.lower() == "default":
        args["lora_target_modules"] = None
    else:
        # Parse input string to list
        args["lora_target_modules"] = args["lora_target_modules"].split(",")

    if args["precision"] in ["bf16", "bf16_autocast", "bf16_buffers_autocast"] and not torch.cuda.is_bf16_supported():
        raise ValueError('Current device does not support bfloat16')

    # Set no_sync if using cpu_offload and gradient accumulation. Turn off if not using gradient accumulation
    if args["use_cpu_offload"] and args["gradient_accumulation_steps"] > 1:
        args["no_sync"] = True
    elif args["no_sync"] and args["gradient_accumulation_steps"] == 1:
        args["no_sync"] = False

    if args["train_type"] in ["hqq_lora"] and HQQLinear is None:
        raise ValueError("HQQ is required to train with `train_type='hqq_lora'`. See ReadMe for details.")

    # Run
    mp.spawn(fsdp_main,
        args = (world_size, args),
        nprocs = torch.cuda.device_count(),
        join = True) 