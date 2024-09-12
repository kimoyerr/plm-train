# Modified From: https://github.com/pytorch/torchtitan/blob/main/torchtitan/train.py


import os
import contextlib
import gc
from collections import defaultdict
from typing import Any, Dict, List
from timeit import default_timer as timer
from dataclasses import asdict

import numpy as np
import torch
from torch.distributed.tensor.parallel import loss_parallel
from torch.distributed.checkpoint.stateful import Stateful
from transformers import EsmTokenizer
import mlflow

from torchtitan.config_manager import JobConfig
from torchtitan.logging_utils import init_logger, logger
from torchtitan.lr_scheduling import get_lr_schedulers
from torchtitan.metrics import build_gpu_memory_monitor, build_metric_logger
from torchtitan.profiling import maybe_enable_profiling
from torchtitan.checkpoint import CheckpointManager

from torchtitan.utils import (
    Color,
    dist_max,
    dist_mean,
    get_num_flop_per_token,
    get_num_params,
    get_peak_flops,
    init_distributed,
    NoColor,
    set_pg_timeouts,
)


from torchtitan.parallelisms import (
    ParallelDims,
)

from train_utils import TrainState, build_optimizer, loss_fn, calc_f1_score
from data_utils import split_fasta_to_json, build_hf_data_loader
from model import ModelArgs, Transformer
from plm_tokenizers import CatalyticTokenizer

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


# MLFlow tracking/logging
# Set the experiment via environment variables
os.environ["MLFLOW_EXPERIMENT_NAME"] = "multi-modal"


# Job config
job_config = JobConfig()
config_file = '/workspace/plm-train/configs/plm_multimodal_llama2_7b.toml'
try:
    with open(config_file, "rb") as f:
        args_dict = defaultdict(defaultdict)
        for k, v in tomllib.load(f).items():
            print(k, v)
            # to prevent overwrite of non-specified keys
            args_dict[k] |= v
except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
    logger.exception(
        f"Error while loading the configuration file: {config_file}"
    )
    logger.exception(f"Error details: {str(e)}")
    raise e

for k, v in args_dict.items():
    class_type = type(k.title(), (), v)
    setattr(job_config, k, class_type())
job_config._validate_config()

init_logger()
logger.info(f"Starting job: {job_config.job.description}")

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


# used for colorful printing
color = Color if job_config.metrics.enable_color_printing else NoColor

# take control of garbage collection to avoid stragglers
_gc_freq = job_config.training.gc_freq
gc.disable()
gc.collect(1)

# Env variables for torch distributed
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

# Cuda device
# init world mesh
world_size = int(os.environ["WORLD_SIZE"])
parallel_dims = ParallelDims(
    dp=job_config.training.data_parallel_degree,
    tp=job_config.training.tensor_parallel_degree,
    pp=job_config.experimental.pipeline_parallel_degree,
    world_size=world_size,
    enable_loss_parallel=job_config.training.enable_loss_parallel,
)
device = torch.device(f"cuda:0")
torch.cuda.set_device(device)

init_distributed(job_config)

# take control of garbage collection to avoid stragglers
# _gc_freq = job_config.training.gc_freq
# gc.disable()
# gc.collect(1)

# Prepare data
data_dir = '/workspace/plm-train/data'
model_inputs_dir = '/workspace/plm-train/data/model_inputs'

# Create training data
train_seq_dataset_name = "uni14230_sequences"
train_catalytic_dataset_name = "uni14230_catalytic_sites"
train_datasets = {}
train_datasets["sequence"] = os.path.join(data_dir, f"{train_seq_dataset_name}.fasta")
train_datasets["catalytic_sites"] = os.path.join(data_dir, f"{train_catalytic_dataset_name}.fasta")
num_training_seq, _ = split_fasta_to_json(data_dir, train_datasets, model_inputs_dir, num_training_seq="all", num_validation_seq=0)

# Create validation
val_seq_dataset_name = "uni3175_sequences"
val_catalytic_dataset_name = "uni3175_catalytic_sites"
val_datasets = {}
val_datasets["sequence"] = os.path.join(data_dir, f"{val_seq_dataset_name}.fasta")
val_datasets["catalytic_sites"] = os.path.join(data_dir, f"{val_catalytic_dataset_name}.fasta")
_, num_validation_seq = split_fasta_to_json(data_dir, val_datasets, model_inputs_dir, num_training_seq=0, num_validation_seq="all")

data_tag_dict = {
    "seq_dataset_name": train_seq_dataset_name,  # "uni3175_sequences"
    "catalytic_dataset_name": train_catalytic_dataset_name,
    "num_training_seq": num_training_seq,
    "num_validation_seq": num_validation_seq,
}

# Create huggingface dataloader
model_seed = 1
model_name = f'facebook/esm1v_t33_650M_UR90S_{model_seed}'
seq_tokenizer = EsmTokenizer.from_pretrained(model_name)
catalytic_tokenizer = CatalyticTokenizer()
# tokenizer.add_tokens(['<bos>', '<eos>'])
data_path = os.path.join(data_dir, 'csv')
data_loader = build_hf_data_loader('ecoli_protein_train', model_inputs_dir, "train", seq_tokenizer, catalytic_tokenizer, batch_size=job_config.training.batch_size, seq_len=256, world_size=1, rank=0, infinite=True)
val_data_loader = build_hf_data_loader('ecoli_protein_val', model_inputs_dir, "validation", seq_tokenizer, catalytic_tokenizer, batch_size=job_config.training.val_batch_size, seq_len=256, world_size=1, rank=0, infinite=True)
# catalytic_train_data_loader = build_hf_data_loader('ecoli_protein_train', json_dirs["catalytic_folder"], "train", catalytic_tokenizer, batch_size=job_config.training.batch_size, seq_len=256, world_size=1, rank=0, infinite=True)
# catalytic_val_data_loader = build_hf_data_loader('ecoli_protein_val', json_dirs["catalytic_folder"], "validation", catalytic_tokenizer, batch_size=job_config.training.val_batch_size, seq_len=256, world_size=1, rank=0, infinite=True)

dataloader_tag_dict = {
    "tokenizer": "facebook/esm1v_t33_650M_UR90S_1",
    "batch_size": 8,
    "seq_len": 256,
}


# Model
model_config = ModelArgs(dim=1024, n_layers=16, n_heads=8)
model_seed = 1
tokenizer = EsmTokenizer.from_pretrained(model_name)
model_config.vocab_size = len(tokenizer.all_tokens)
model_config.catalytic_vocab_size = len(catalytic_tokenizer.all_tokens)
model_config.max_seq_len = 256
model_config.final_output = "catalytic-sites"
assert model_config.final_output == "catalytic-sites"  # To raise an error if the final output is not catalytic-sites 

# Positional embeddings
model_config.pos_emb_type = "cope"

# Class labels for catalytic-sites
if model_config.final_output == "catalytic-sites":  
    # Get the decoded labels
    catalytic_labels = catalytic_tokenizer.all_tokens
    catalytic_site_index = catalytic_labels.index("X")
    # Assign 1000 weight to catalytic site and 1 to all other classes
    class_weights = [10 if i == catalytic_site_index else 1 for i in range(10)]
    # Normalize to 0 to 1
    class_weights = [float(i)/sum(class_weights) for i in class_weights]
    class_weights = torch.tensor(class_weights)
else:
    class_weights = None

# TODO: RoPE positional encoding variations
# TODO: Identical word probing to understand the effect of positional encoding: https://arxiv.org/pdf/2310.12864
# TODO: Context aware positional embeddings (CoPE): https://arxiv.org/abs/2405.18719
# TODO: have 2 positional encodings, one for the immediate neighbors reflecting peptide bonds between adjacent residues and hydrogen bonds between residues separated by 3 or 4 residues
model = Transformer(model_config)
# model.to_empty(device="cuda")
model.to(device="cuda:0")

# Convert model_config dataclass to dictionary
model_config_dict = asdict(model_config)

# loss_parallel enables dispatching to efficient loss operators
loss_parallel_ctx = (
    loss_parallel if parallel_dims.loss_parallel_enabled else contextlib.nullcontext
)

# Wandb Logger
wandb_logger =  Logger(job_config, log_to="wandb", project_name="multi-modal",
                    name="multi-modal-predict-catalytic", rank=0)

# build optimizer after applying parallelisms to the model
optimizers = [build_optimizer(model, job_config)]
schedulers = get_lr_schedulers(optimizers, job_config)
metric_logger = build_metric_logger(job_config)

train_state = TrainState()
model.train()

checkpoint = CheckpointManager(
    model_parts=[model],
    optimizers=optimizers,
    lr_schedulers=schedulers,
    dataloader=data_loader,
    states={"train_state": train_state},
    job_config=job_config,
)

job_config.checkpoint.create_seed_checkpoint = True
if job_config.checkpoint.create_seed_checkpoint:
    assert (
        world_size == 1
    ), "Must create seed-checkpoint using one gpu, to disable sharding"
    checkpoint.save(curr_step=0, force=True)
    logger.info("Created seed checkpoint")
    

checkpoint_loaded = checkpoint.load()
if parallel_dims.pp_enabled and not checkpoint_loaded:
    raise RuntimeError(
        "Pipeline Parallelism requires meta-initialization and loading seed checkpoint. "
        "Please run `./create_seed_checkpoint.sh` and rerun training with `--checkpoint.enable_checkpoint`"
    )

data_iterator = iter(data_loader)
batch = next(data_iterator)
input_ids, labels = batch
# Convert list of tensors to tensor
if len(input_ids) > 1:
    input_ids = torch.stack(input_ids, dim=1)
    labels = torch.stack(labels, dim=1)
input_ids = input_ids.cuda()
labels = labels.cuda()
preds = model(input_ids)
if len(labels.shape) > 2:
    loss = loss_fn(preds, labels[:, 1, :], class_weights=class_weights, reduction="mean")
else:
    loss = loss_fn(preds, labels, class_weights=class_weights)
print(loss)

# Training iteration
# plot losses loaded from checkpoint (if any) to TensorBoard
# NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
#       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
if train_state.step > 0:
    for idx, step in enumerate(train_state.log_steps):
        metrics = {
            "loss_metrics/global_avg_loss": train_state.global_avg_losses[idx],
            "loss_metrics/global_max_loss": train_state.global_max_losses[idx],
        }
        metric_logger.log(metrics, step=step)

logger.info(f"Training starts at step {train_state.step + 1}")
with maybe_enable_profiling(
    job_config, global_step=train_state.step
) as torch_profiler:
    
    with mlflow.start_run():
        # Set mlflow tags
        mlflow.set_tags(data_tag_dict)
        mlflow.set_tags(dataloader_tag_dict)
        mlflow.set_tags(model_config_dict)
        
        checkpoint.reset()

        # variables used to keep info for metrics logging
        losses_since_last_log: List[float] = []
        ntokens_since_last_log = 0
        data_loading_times: List[float] = []
        time_last_log = timer()
        # gpu_memory_monitor.reset_peak_stats()

        val_loss = torch.tensor(100000)  # Initial loss for logging purposes
        val_perplexity = torch.tensor(100000)  # Initial loss for logging purposes
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            if train_state.step > 1 and train_state.step % _gc_freq == 0:
                gc.collect(1)

            # get batch
            data_load_start = timer()
            batch = next(data_iterator)
            input_ids, labels = batch
            if len(input_ids) > 1:
                input_ids = torch.stack(input_ids, dim=1)
                labels = torch.stack(labels, dim=1)
                input_ids = input_ids.cuda()
                labels = labels.cuda()
                loss = loss_fn(preds, labels[:, 1, :], class_weights=class_weights, reduction="mean")
            else:
                loss = loss_fn(preds, labels)
            ntokens_since_last_log += labels.numel()
            data_loading_times.append(timer() - data_load_start)

            input_ids = input_ids.cuda()
            labels = labels.cuda()
            optimizers[0].zero_grad()

            with loss_parallel_ctx():
                pred = model(input_ids)
                if len(labels) > 1:
                    loss = loss_fn(pred, labels[:, 1, :], class_weights=class_weights, reduction="mean")
                else:
                    loss = loss_fn(pred, labels, class_weights=class_weights, reduction="mean")

                loss.backward()

            # Validation loss

            if train_state.step % job_config.metrics.validation_freq == 0:
                mlflow.pytorch.log_model(model, "model")
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    num_val_seq_run = 0
                    num_val_batches = 0
                    # Keep track of the number of times the val data loader is looped over as a
                    # Store all val_preds and labels
                    all_val_preds = []
                    all_val_labels = []
                    prev_num_repeats = val_data_loader.dataset.num_repeats
                    for val_batch in val_data_loader:
                        if val_data_loader.dataset.num_repeats > prev_num_repeats:
                            break
                        val_input_ids, val_labels = val_batch
                        if len(val_input_ids) > 1:
                            val_input_ids = torch.stack(val_input_ids, dim=1)
                            val_labels = torch.stack(val_labels, dim=1)
                            
                        val_input_ids = val_input_ids.cuda()
                        val_labels = val_labels.cuda()
                        val_preds = model(val_input_ids)
                        all_val_preds.append(val_preds.cpu())
                        if len(val_labels.shape) > 2:
                            all_val_labels.append(val_labels[:, 1, :].cpu())
                            tmp_val_loss = loss_fn(val_preds, val_labels[:, 1, :], class_weights=class_weights, reduction="mean")
                        else:
                            tmp_val_loss = loss_fn(val_preds, val_labels, class_weights=class_weights, reduction="mean")
                        val_loss += tmp_val_loss
                        num_val_seq_run += len(val_labels)
                        num_val_batches += 1
                val_loss /= num_val_batches
                
                # F1-score
                # Convert list of tensors to tensor by concatenating along the first dimension
                all_val_preds = torch.cat(all_val_preds, dim=0)
                all_val_labels = torch.cat(all_val_labels, dim=0)
                pos_label = catalytic_labels.index("X")
                neg_label = catalytic_labels.index("Z")
                (
                    val_f1_score, 
                    val_precision, 
                    val_sensitivity, 
                    val_true_positives, 
                    val_false_positives, 
                    val_true_negatives, 
                    val_false_negatives,
                ) = calc_f1_score(all_val_preds, all_val_labels, pos_label, neg_label)

                mlflow.log_metric("mean_val_loss", val_loss, step=train_state.step)
                wandb_logger.log({"mean_val_loss": val_loss, "step": train_state.step}, rank=0)
                val_perplexity = torch.exp(val_loss)
                mlflow.log_metric("val_perplexity", val_perplexity, step=train_state.step)
                wandb_logger.log({"val_perplexity": val_perplexity, "step": train_state.step}, rank=0)
                wandb_logger.log({"val_F1_scores": val_f1_score, "step": train_state.step}, rank=0)
                wandb_logger.log({"val_precision": val_precision, "step": train_state.step}, rank=0)
                wandb_logger.log({"val_sensitivity": val_sensitivity, "step": train_state.step}, rank=0)
                wandb_logger.log({"val_true_positives": val_true_positives, "step": train_state.step}, rank=0)
                wandb_logger.log({"val_false_positives": val_false_positives, "step": train_state.step}, rank=0)
                wandb_logger.log({"val_true_negatives": val_true_negatives, "step": train_state.step}, rank=0)
                wandb_logger.log({"val_false_negatives": val_false_negatives, "step": train_state.step}, rank=0)

                gc.collect(1)
                model.train()
        
            # clip gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), job_config.training.max_norm, foreach=True
            )

            # optimizer step
            checkpoint.wait_for_staging()
            optimizers[0].step()
            schedulers.step()

            losses_since_last_log.append(loss)
            mlflow.log_metric("train_batches_steps", train_state.step, step=train_state.step)
            mlflow.log_metric("train_loss", loss, step=train_state.step)
            wandb_logger.log({"train_loss": loss, "step":train_state.step}, rank=0)
            
            perplexity = torch.exp(loss)
            mlflow.log_metric("train_perplexity", perplexity, step=train_state.step)
            wandb_logger.log({"train_perplexity": perplexity, "step":train_state.step}, rank=0)
            print(perplexity.item())

            # log metrics
            if (
                train_state.step == 1
                or train_state.step % job_config.metrics.log_freq == 0
            ):
                losses = [loss.item() for loss in losses_since_last_log]
                avg_loss, max_loss = (
                    np.mean(losses),
                    np.max(losses),
                )
                if parallel_dims.dp_enabled:
                    global_avg_loss, global_max_loss = (
                        dist_mean(avg_loss, dp_mesh).item(),
                        dist_max(max_loss, dp_mesh).item(),
                    )
                else:
                    global_avg_loss, global_max_loss = avg_loss, max_loss

                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                time_delta = timer() - time_last_log

                # tokens per second, abbr. as wps by convention
                wps = ntokens_since_last_log / (
                    time_delta * parallel_dims.model_parallel_size
                )
                # model FLOPS utilization
                # For its definition and calculation, please refer to the PaLM paper:
                # https://arxiv.org/abs/2204.02311
                # mfu = 100 * num_flop_per_token * wps / gpu_peak_flops

                time_end_to_end = time_delta / job_config.metrics.log_freq
                time_data_loading = np.mean(data_loading_times)
                time_data_loading_pct = 100 * np.sum(data_loading_times) / time_delta

                # gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                metrics = {
                    "loss_metrics/global_avg_loss": global_avg_loss,
                    "loss_metrics/global_max_loss": global_max_loss,
                    "wps": wps,
                    # "mfu(%)": mfu,
                    "time_metrics/end_to_end(s)": time_end_to_end,
                    "time_metrics/data_loading(s)": time_data_loading,
                    "time_metrics/data_loading(%)": time_data_loading_pct,
                    # "memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
                    # "memory/max_active(%)": gpu_mem_stats.max_active_pct,
                    # "memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
                    # "memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
                    # "memory/num_alloc_retries": gpu_mem_stats.num_alloc_retries,
                    # "memory/num_ooms": gpu_mem_stats.num_ooms,
                }
                metric_logger.log(metrics, step=train_state.step)

                logger.info(
                    f"{color.cyan}step: {train_state.step:2}  "
                    f"{color.yellow}loss: {global_avg_loss:7.4f}  "
                    f"{color.green}val_loss: {val_loss:7.4f}  "
                    # f"{color.yellow}memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
                    # f"({gpu_mem_stats.max_reserved_pct:.2f}%)  "
                    f"{color.blue}wps: {round(wps):,}  "
                    # f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
                )

                losses_since_last_log.clear()
                ntokens_since_last_log = 0
                data_loading_times.clear()
                time_last_log = timer()
                # gpu_memory_monitor.reset_peak_stats()
            
            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            )