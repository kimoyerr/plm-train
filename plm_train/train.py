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
from torchtitan.lr_scheduling import get_lr_scheduler
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


from fasta_utils import fasta_to_json
from train_utils import TrainState, build_optimizer, loss_fn
from data_utils import build_hf_data_loader
from model import ModelArgs, Transformer

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


# MLFlow tracking/logging
# Set the experiment via environment variables
os.environ["MLFLOW_EXPERIMENT_NAME"] = "CoPE"


# Job config
job_config = JobConfig()
config_file = '/workspace/plm-train/configs/plm_llama2_7b.toml'
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
dataset_name = "uniprotkb_E_coli_AND_model_organism_833_2024_05_04"
input_fasta = os.path.join(data_dir, f"{dataset_name}.fasta")
train_json = os.path.join(data_dir, f"csv/{dataset_name}_train.json")
val_json = os.path.join(data_dir, f"csv/{dataset_name}_val.json")
num_training_seq = 3000
num_validation_seq = 1000
random_seed = 42
fasta_to_json(input_fasta, train_json, val_json, ntrain_seq=num_training_seq, nval_seq=num_validation_seq, random_seed=random_seed)

data_tag_dict = {
    "dataset_name": dataset_name,
    "num_training_seq": num_training_seq,
    "num_validation_seq": num_validation_seq,
    "random_seed_train_val": random_seed,
}

# Create huggingface dataloader
model_seed = 1
model_name = f'facebook/esm1v_t33_650M_UR90S_{model_seed}'
tokenizer = EsmTokenizer.from_pretrained(model_name)
# tokenizer.add_tokens(['<bos>', '<eos>'])
data_path = os.path.join(data_dir, 'csv')
data_loader = build_hf_data_loader('ecoli_protein_train', data_path, "train", tokenizer, batch_size=job_config.training.batch_size, seq_len=256, world_size=1, rank=0, infinite=True)
val_data_loader = build_hf_data_loader('ecoli_protein_val', data_path, "validation", tokenizer, batch_size=job_config.training.val_batch_size, seq_len=256, world_size=1, rank=0, infinite=True)
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
model_config.max_seq_len = 256

# Positional embeddings
model_config.pos_emb_type = "cope"

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

# build optimizer after applying parallelisms to the model
optimizer = build_optimizer(model, job_config)
scheduler = get_lr_scheduler(optimizer, job_config)
metric_logger = build_metric_logger(job_config)

train_state = TrainState()
model.train()

checkpoint = CheckpointManager(
    model=model,
    optimizer=optimizer,
    lr_scheduler=scheduler,
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
input_ids = input_ids.cuda()
labels = labels.cuda()
preds = model(input_ids)
loss = loss_fn(preds, labels)
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
            ntokens_since_last_log += labels.numel()
            data_loading_times.append(timer() - data_load_start)

            input_ids = input_ids.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()

            with loss_parallel_ctx():
                pred = model(input_ids)
                loss = loss_fn(pred, labels, reduction="mean")
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
                    prev_num_repeats = val_data_loader.dataset.num_repeats
                    for val_batch in val_data_loader:
                        if val_data_loader.dataset.num_repeats > prev_num_repeats:
                            break
                        val_input_ids, val_labels = val_batch
                        val_input_ids = val_input_ids.cuda()
                        val_labels = val_labels.cuda()
                        val_preds = model(val_input_ids)
                        val_loss += loss_fn(val_preds, val_labels, reduction="mean")
                        num_val_seq_run += len(val_labels)
                        num_val_batches += 1
                val_loss /= num_val_batches
                mlflow.log_metric("mean_val_loss", val_loss, step=train_state.step)
                val_perplexity = torch.exp(val_loss)
                mlflow.log_metric("val_perplexity", val_perplexity, step=train_state.step)
                gc.collect(1)
                model.train()
        
            # clip gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), job_config.training.max_norm, foreach=True
            )

            # optimizer step
            checkpoint.wait_for_staging()
            optimizer.step()
            scheduler.step()

            losses_since_last_log.append(loss)
            mlflow.log_metric("train_batches_steps", train_state.step, step=train_state.step)
            mlflow.log_metric("train_loss", loss, step=train_state.step)
            perplexity = torch.exp(loss)
            mlflow.log_metric("train_perplexity", perplexity, step=train_state.step)
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