# General
import torch, os, gc, time, safetensors, copy, math, types
import re
import functools
from typing import List, Dict
from collections import OrderedDict
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import get_linear_schedule_with_warmup
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import nullcontext
from safetensors.torch import save_file
from tqdm.auto import tqdm
# from typing import List, Dict
import pandas as pd
from safetensors import safe_open

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
# from bitsandbytes.nn import Linear4bit, Params4bit
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
from finetune import get_confit_dataloader, fsdp_auto_wrap_policy_confit

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





#############################################################################################################################################

def load_lora_weights(safetensor_paths):
    # Initialize an empty dictionary to store the tensors
    lora_tensors = {}
    full_tensors = {}

    # Open the .safetensors paths and load the tensors
    for key, value in safetensor_paths.items():
        with safe_open(value, framework="pt", device=0) as f:
            lora_tensors[key] = {}
            full_tensors[key] = {}
            for k in f.keys():
                if "lora" in k:
                    # print(k)
                    lora_tensors[key][k] = f.get_tensor(k)
                    # Get the corresponding full tensor
                    if "lora_A.default" in k:
                        full_tensor_name = k.replace("lora_A.default", "base_layer")
                        # print(full_tensor_name)
                        full_tensors[key][full_tensor_name] = f.get_tensor(full_tensor_name)
                else:
                    full_tensors[key][k] = f.get_tensor(k)
    
    return full_tensors, lora_tensors


def weight_param_dict_to_vectors(lora_weights):
    all_task_vectors = []
    for task_vector in lora_weights.values():
        task_vector_param_dict = copy.deepcopy(task_vector)
        sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))

        # Tensor, shape (num_total_params, )
        all_task_vectors.append(nn.utils.parameters_to_vector([param.flatten() for param in sorted_task_vector_param_dict.values()]))
    
    return all_task_vectors


# From https://github.com/yule-BUAA/MergeLM/tree/main
def mask_smallest_magnitude_param_values(flattened_models_to_merge_param, param_value_mask_rate=0.8):

            # num_models_to_merge, num_total_params = flattened_models_to_merge_param.shape
            num_mask_params = int(flattened_models_to_merge_param.shape[1] * param_value_mask_rate)

            # Tensor, shape (num_models_to_merge, 1), find the num_mask_params-th smallest magnitude element of all the parameters in each individual model
            kth_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=num_mask_params, dim=1, keepdim=True)
            # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
            mask = flattened_models_to_merge_param.abs() >= kth_values

            return flattened_models_to_merge_param * mask


# From https://github.com/yule-BUAA/MergeLM/tree/main
def get_param_signs(flattened_models_to_merge_param):
    # Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
    param_signs = torch.sign(flattened_models_to_merge_param.sum(dim=0))
    # Tensor, shape (, ), a scalar, replace 0 in param_signs to the major sign in param_signs
    majority_sign = torch.sign(param_signs.sum(dim=0))
    param_signs[param_signs == 0] = majority_sign
    
    return param_signs


# From https://github.com/yule-BUAA/MergeLM/tree/main
def disjoint_merge(flattened_models_to_merge_param, param_signs):
    """
    disjoint merge that only keeps the parameter values in individual models whose signs are the same as the param_signs, and calculates the averaged parameters.
    """
    # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
    param_to_preserve_mask = ((param_signs.unsqueeze(dim=0) > 0) & (flattened_models_to_merge_param > 0)) | ((param_signs.unsqueeze(dim=0) < 0) & (flattened_models_to_merge_param < 0))
    # Tensor, shape (num_models_to_merge, num_total_params), the preserved parameters
    param_to_preserve = flattened_models_to_merge_param * param_to_preserve_mask

    # Tensor, shape (num_total_params, ), the number of models whose parameters can be preserved
    num_models_param_preserved = (param_to_preserve != 0).sum(dim=0).float()
    # Tensor, shape (num_total_params, ), the averaged flattened parameters
    merged_flattened_param = torch.sum(param_to_preserve, dim=0) / torch.clamp(num_models_param_preserved, min=1.0)

    return merged_flattened_param


def save_weights(full_weights, lora_weights, output_path):
    # Add the full weights to the avg_lora_weights dictionary
    lora_weights.update(full_weights)

    # Save the avg_lora_weights to a file
    save_file(lora_weights, output_path)


def average_lora_weights(full_weights, lora_weights, output_path):
    # Iterate over the lora_weights and average the tensors
    # Initialize a dictionary to store the sums and counts for each weight
    sums_counts = {}

    # Iterate over the top-level keys
    for data_key, sub_dict in lora_weights.items():
        for weight_key, value in sub_dict.items():
            if weight_key not in sums_counts:
                sums_counts[weight_key] = {'sum': 0, 'count': 0}
            sums_counts[weight_key]['sum'] += value
            sums_counts[weight_key]['count'] += 1

    # Calculate the averages
    avg_lora_weights = {weight_key: sums_counts[weight_key]['sum'] / sums_counts[weight_key]['count'] for weight_key in sums_counts}

    # Get the full weights for the first key. Since all the full weights are the same, we can just use the first key
    full_weights = full_weights[list(full_weights.keys())[0]]
    # Add the full weights to the avg_lora_weights dictionary
    avg_lora_weights.update(full_weights)

    # Save the avg_lora_weights to a file
    save_file(avg_lora_weights, output_path)

    return avg_lora_weights


def mixlora_main(local_rank, world_size, args):

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

    # Logger
    logger = Logger(args, log_to=args["log_to"], project_name=args["project_name"],
                entity=args["entity"], group=args["group"], name=args["name"], rank=rank)

    # Load the Peft model
    # model_path = "model_outputs/Q837P4_ENTFA_Meier_2023/lora_dense/train_288_shot/model_state_dict_epoch_8.safetensors"
    model_path = args["weights_path"]
    lora_tensors = {}
    full_tensors = {}
    epoch = 6


    # Load model
    model_name = "facebook/esm1v_t33_650M_UR90S_1"
    torch_dtype, compute_dtype = torch.float32, torch.float16
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        use_cache=False,
        torch_dtype=torch_dtype,
        _attn_implementation="eager"

    )
    peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False,
                r=8,
                lora_alpha=8,
                lora_dropout=0.1,
                target_modules=["dense"],
                use_dora=0,
            )

    model = get_peft_model(model, peft_config)
    
    missing_keys = safetensors.torch.load_model(model, model_path, strict=False)
    print(missing_keys)  # decoder.weight is expected to be missing. That is okay

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm1v_t33_650M_UR90S_1")
    tokenizer.pad_token_id = tokenizer.eos_token_id # TODO check if it exists first

    # FSDP
    sharding_strategy = "full_shard"
    my_auto_wrap_policy = fsdp_auto_wrap_policy_confit(model)
    mp_policy = None
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


    # Synchronize at the start
    dist.barrier()

    # Perform inference using this model
    model.eval()
    # Test
    print(f"Testing")
    data_args = {
        "seed": 42, 
        "batch_size": 4,
        "protein_dataset": args["protein_dataset"],
        "protein_trainset_path": args["protein_trainset_path"],
        "protein_testset_path": args["protein_testset_path"],
        "protein_valset_path": args["protein_valset_path"],
    }
    _, valloader, testloader = get_confit_dataloader(data_args)

    # Evaluate
    with torch.no_grad():
        model.eval()
        # for data in testloader:
        #     print(data.device)
        sr = evaluate(model, valloader, tokenizer, accelerator="fsdp")
        print(f'========epoch{epoch}; val spearman correlation :{sr}=================')
        logger.log({"val_spearman": sr}, rank)
        sr = evaluate(model, testloader, tokenizer, accelerator="fsdp")
        print(f'========epoch{epoch}; test spearman correlation :{sr}=================')
        logger.log({"test_spearman": sr}, rank)


@call_parse()
def main():
    # Run
    world_size = torch.cuda.device_count()
    # Get all args which will be passed to fsdp_main
    args = {}
    args["master_addr"] = "localhost"
    args["master_port"] = "12355"
    args["use_cpu_offload"] = True
    args["low_memory"] = False
    args["log_to"] = "wandb"
    args["project_name"] = "fsdp_qlora_confit_fewshot_finetuning"
    args["protein_dataset"] = "AMIE_PSEAE_Wrenbeck_2017"
    args["group"] = args["protein_dataset"]
    args["protein_trainset_path"] = "/workspace/ConFit/data/proteingym/AMIE_PSEAE_Wrenbeck_2017/train_288_shot.csv"
    args["protein_testset_path"] = "/workspace/ConFit/data/proteingym/AMIE_PSEAE_Wrenbeck_2017/test.csv"
    args["protein_valset_path"] = "/workspace/ConFit/data/proteingym/AMIE_PSEAE_Wrenbeck_2017/val_288_shot.csv"
    args["name"] = None
    args['entity'] = None

    ckpt_files = {
        "A0A1I9GEU1_NEIME_Kennouche_2019": "/workspace/plm-train/model_outputs/A0A1I9GEU1_NEIME_Kennouche_2019/lora_dense/train_288_shot/model_state_dict_epoch_4.safetensors",
        "AMIE_PSEAE_Wrenbeck_2017": "/workspace/plm-train/model_outputs/AMIE_PSEAE_Wrenbeck_2017/lora_dense/train_288_shot/model_state_dict_epoch_4.safetensors",
        "Q837P4_ENTFA_Meier_2023": "/workspace/plm-train/model_outputs/Q837P4_ENTFA_Meier_2023/lora_dense/train_288_shot/model_state_dict_epoch_8.safetensors",
        "Q837P5_ENTFA_Meier_2023": "/workspace/plm-train/model_outputs/Q837P5_ENTFA_Meier_2023/lora_dense/train_288_shot/model_state_dict_epoch_2.safetensors",
        "Q59976_STRSQ_Romero_2015": "/workspace/plm-train/model_outputs/Q59976_STRSQ_Romero_2015/lora_dense/train_288_shot/model_state_dict_epoch_2.safetensors",
        "RNC_ECOLI_Weeks_2023": "/workspace/plm-train/model_outputs/RNC_ECOLI_Weeks_2023/lora_dense/train_288_shot/model_state_dict_epoch_3.safetensors",
        "RPC1_LAMBD_Li_2019_high-expression": "/workspace/plm-train/model_outputs/RPC1_LAMBD_Li_2019_high-expression/lora_dense/train_192_shot/model_state_dict_epoch_2.safetensors",
    }
    args["lora_weights"] = ckpt_files
    print(f"Loading LORA weights")
    full_weights, lora_weights = load_lora_weights(ckpt_files)
    print(f"Finished loading LORA weights")

    # args["weights_path"] = "/workspace/plm-train/notebooks/data/average_lora_weights.safetensors"
    args["weights_path"] = "/workspace/plm-train/notebooks/data/TIES_lora_weights.safetensors"

    # TIES merging
    flattened_models_to_merge_param = weight_param_dict_to_vectors(lora_weights)
    flattened_models_to_merge_param = torch.vstack(flattened_models_to_merge_param)
    with torch.no_grad():
            # Tensor, shape (num_models_to_merge, num_total_params), mask the smallest-magnitude parameter values using param_value_mask_rate
            flattened_models_to_merge_param = mask_smallest_magnitude_param_values(flattened_models_to_merge_param=flattened_models_to_merge_param, param_value_mask_rate=0.8)
            # Tensor, shape (num_total_params, ), get the signs for each parameter in flattened_models_to_merge_param
            param_signs = get_param_signs(flattened_models_to_merge_param)
            # Tensor, shape (num_total_params, ), disjoint merge
            merged_flattened_param = disjoint_merge(flattened_models_to_merge_param=flattened_models_to_merge_param, param_signs=param_signs)
            # Get the first task vector param dict
            task_vector_param_dict = copy.deepcopy(lora_weights[list(lora_weights.keys())[0]])
            sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))
            # Update the sorted task vector param dict weights with the merged weights
            nn.utils.vector_to_parameters(merged_flattened_param, sorted_task_vector_param_dict.values())
            # Save the merged weights
            # Get the full weights for the first key. Since all the full weights are the same, we can just use the first key
            sel_full_weights = full_weights[list(full_weights.keys())[0]]
            save_weights(sel_full_weights, sorted_task_vector_param_dict, args["weights_path"])

    # Average the lora weights
    # average_lora_weights(full_weights, lora_weights, args["weights_path"])

    mp.spawn(mixlora_main,
        args = (world_size, args),
        nprocs = torch.cuda.device_count(),
        join = True) 