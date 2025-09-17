# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Inference codes adapted from [SeedVR]
# https://github.com/ByteDance-Seed/SeedVR/blob/main/projects/inference_seedvr2_7b.py

import math
import os
import gc
import random
import sys
import mediapy
import torch
import time
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig, OmegaConf
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
)
from common.distributed import (
    get_device,
    get_global_rank,
    get_local_rank,
    meta_param_init_fn,
    meta_non_persistent_buffer_init_fn,
    init_torch,
)
from common.distributed.advanced import (
    init_unified_parallel,
    get_unified_parallel_world_size,
    get_sequence_parallel_rank,
    init_model_shard_cpu_group,
)
from common.logger import get_logger
from common.config import create_object
from common.distributed import get_device, get_global_rank
from torchvision.transforms import Compose, Normalize, ToTensor
from humo.models.wan_modules.t5 import T5EncoderModel
from humo.models.wan_modules.vae import WanVAE
from humo.models.utils.utils import tensor_to_video, prepare_json_dataset
from contextlib import contextmanager
import torch.cuda.amp as amp
from humo.models.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from humo.utils.audio_processor_whisper import AudioProcessor
from humo.utils.wav2vec import linear_interpolation_fps


image_transform = Compose([
    ToTensor(),
    Normalize(mean=0.5, std=0.5),
])

SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
    '1024*1024': (1024, 1024),
}

def clever_format(nums, format="%.2f"):
    from typing import Iterable
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []
    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)

    return clever_nums



# ==============================================================================
# GPU Monitoring and Progress Utilities for HuMo
# ==============================================================================

def get_gpu_memory_info():
    """Get current GPU memory usage information."""
    if not torch.cuda.is_available():
        return {"available": 0, "total": 0, "used": 0, "free": 0}
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free_memory = total_memory - reserved
    
    return {
        "device": device,
        "total": total_memory / (1024**3),  # GB
        "allocated": allocated / (1024**3),  # GB
        "reserved": reserved / (1024**3),   # GB
        "free": free_memory / (1024**3)     # GB
    }

def log_multi_gpu_utilization(prefix="GPU"):
    """Log current GPU utilization and memory usage."""
    if not torch.cuda.is_available():
        print(f"[{prefix}] CUDA not available")
        return
    
    mem_info = get_gpu_memory_info()
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    
    print(f"[{prefix}] Device: {device_name} (GPU {current_device}/{device_count-1})")
    print(f"[{prefix}] Memory: {mem_info['allocated']:.1f}GB/{mem_info['total']:.1f}GB "
          f"({100 * mem_info['allocated'] / mem_info['total']:.1f}% allocated)")

def format_time_remaining(avg_step_time, steps_done, total_steps):
    """Format estimated time remaining based on average step duration."""
    if steps_done == 0 or avg_step_time == 0:
        return "calculating..."
    
    remaining_steps = total_steps - steps_done
    remaining_seconds = remaining_steps * avg_step_time
    
    if remaining_seconds < 60:
        return f"{remaining_seconds:.0f}s"
    elif remaining_seconds < 3600:
        minutes = remaining_seconds // 60
        seconds = remaining_seconds % 60
        return f"{minutes:.0f}m {seconds:.0f}s"
    else:
        hours = remaining_seconds // 3600
        minutes = (remaining_seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

def format_duration(seconds):
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:.0f}m {secs:.1f}s"
    else:
        hours = seconds // 3600

def log_multi_gpu_utilization(prefix="GPU", show_detailed=False):
    """Production-ready multi-GPU utilization diagnostics for H200 systems."""
    if not torch.cuda.is_available():
        print(f"[{prefix}] CUDA not available")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"[{prefix}] === GPU Utilization Report ({num_gpus} H200s) ===")
    
    total_memory_gb = 0
    total_allocated_gb = 0
    
    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            props = torch.cuda.get_device_properties(gpu_id)
            total_mem = props.total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            free_mem = total_mem - reserved
            
            utilization_pct = (allocated / total_mem) * 100
            
            # Show utilization status
            status = "🔥 ACTIVE" if allocated > 1.0 else "💤 IDLE"
            
            print(f"[{prefix}]   GPU {gpu_id}: {status} - {allocated:.1f}GB/{total_mem:.1f}GB ({utilization_pct:.0f}%)")
            
            if show_detailed:
                print(f"[{prefix}]          Reserved: {reserved:.1f}GB | Free: {free_mem:.1f}GB")
                print(f"[{prefix}]          {props.name} | CC: {props.major}.{props.minor}")
            
            total_memory_gb += total_mem
            total_allocated_gb += allocated
    
    # Overall stats
    overall_util = (total_allocated_gb / total_memory_gb) * 100
    active_gpus = sum(1 for gpu_id in range(num_gpus) 
                     if torch.cuda.memory_allocated(gpu_id) / (1024**3) > 1.0)
    
    print(f"[{prefix}] Total: {total_allocated_gb:.1f}GB/{total_memory_gb:.1f}GB ({overall_util:.0f}%) | Active GPUs: {active_gpus}/{num_gpus}")
    print(f"[{prefix}] {'=' * 50}")

def log_model_distribution_status(model, prefix="MODEL"):
    """Log how model components are distributed across GPUs."""
    if not hasattr(model, '_multi_gpu') or not model._multi_gpu:
        print(f"[{prefix}] Single GPU mode - all components on GPU 0")
        return
    
    print(f"[{prefix}] Multi-GPU Distribution Status:")
    
    # Check embeddings
    embedding_gpu = None
    if hasattr(model, 'patch_embedding') and hasattr(model.patch_embedding, 'device'):
        embedding_gpu = model.patch_embedding.device.index
    
    # Check transformer blocks distribution
    block_distribution = {}
    if hasattr(model, 'blocks'):
        for i, block in enumerate(model.blocks):
            if hasattr(block, 'device'):
                gpu_id = block.device.index
                if gpu_id not in block_distribution:
                    block_distribution[gpu_id] = []
                block_distribution[gpu_id].append(i)
    
    # Check head
    head_gpu = None
    if hasattr(model, 'head') and hasattr(model.head, 'device'):
        head_gpu = model.head.device.index
    
    print(f"[{prefix}]   Embeddings: GPU {embedding_gpu}")
    for gpu_id in sorted(block_distribution.keys()):
        blocks = block_distribution[gpu_id]
        print(f"[{prefix}]   Blocks {min(blocks)}-{max(blocks)}: GPU {gpu_id} ({len(blocks)} blocks)")
    print(f"[{prefix}]   Head: GPU {head_gpu}")

def log_inference_performance_summary(step_times, total_time, num_gpus, prefix="PERF"):
    """Log performance summary focusing on multi-GPU efficiency."""
    if not step_times:
        return
    
    avg_step_time = sum(step_times) / len(step_times)
    min_step_time = min(step_times)
    max_step_time = max(step_times)
    
    # Calculate theoretical vs actual speedup
    single_gpu_estimate = total_time * num_gpus  # Very rough estimate
    efficiency = (single_gpu_estimate / total_time) * 100 if num_gpus > 1 else 100
    
    print(f"[{prefix}] === Performance Summary ===")
    print(f"[{prefix}] Total Time: {format_duration(total_time)}")
    print(f"[{prefix}] Steps: {len(step_times)} | Avg: {avg_step_time:.1f}s | Range: {min_step_time:.1f}s-{max_step_time:.1f}s")
    
    if num_gpus > 1:
        print(f"[{prefix}] Multi-GPU: {num_gpus} GPUs | Est. Efficiency: {efficiency:.0f}%")
        if efficiency < 70:
            print(f"[{prefix}] ⚠️  Low efficiency - check GPU utilization balance")
    
    # Throughput estimation
    fps = len(step_times) / total_time
    print(f"[{prefix}] Throughput: {fps:.1f} inference steps/second")
    print(f"[{prefix}] {'=' * 40}")



class Generator():
    def __init__(self, config: DictConfig):
        self.config = config.copy()
        OmegaConf.set_readonly(self.config, True)
        self.logger = get_logger(self.__class__.__name__)
        
        init_torch(cudnn_benchmark=False)

    def maybe_empty_cache(self):
        """Conditionally free CUDA cache if HUMO_EMPTY_CACHE=1.
        
        Frequent empty_cache() can slow inference by ~10-20% due to
        allocator overhead. Keep opt-in for memory-constrained runs.
        """
        if os.environ.get("HUMO_EMPTY_CACHE", "0") == "1":
            torch.cuda.empty_cache()

    def entrypoint(self):
        self.configure_models()
        self.inference_loop()
    
    def get_fsdp_sharding_config(self, sharding_strategy, device_mesh_config):
        device_mesh = None
        fsdp_strategy = ShardingStrategy[sharding_strategy]
        if (
            fsdp_strategy in [ShardingStrategy._HYBRID_SHARD_ZERO2, ShardingStrategy.HYBRID_SHARD]
            and device_mesh_config is not None
        ):
            device_mesh = init_device_mesh("cuda", tuple(device_mesh_config))
        return device_mesh, fsdp_strategy

    def configure_models(self):
        """Configure all models with detailed timing and GPU monitoring."""
        start_time = time.perf_counter()
        
        print("[MODELS] Starting model configuration...")
        log_multi_gpu_utilization("SETUP")
        
        # DiT Model Loading
        print("[MODELS] Loading DiT model...")
        dit_start = time.perf_counter()
        self.configure_dit_model(device="cpu")
        dit_time = time.perf_counter() - dit_start
        print(f"[MODELS] ✓ DiT model loaded in {format_duration(dit_time)}")
        
        # VAE Model Loading
        print("[MODELS] Loading VAE model...")
        vae_start = time.perf_counter()
        self.configure_vae_model()
        vae_time = time.perf_counter() - vae_start
        print(f"[MODELS] ✓ VAE model loaded in {format_duration(vae_time)}")
        log_multi_gpu_utilization("MODELS-READY")
        
        # Audio Model Loading (if enabled)
        audio_time = 0
        if self.config.generation.get('extract_audio_feat', False):
            print("[MODELS] Loading Wav2Vec audio model...")
            audio_start = time.perf_counter()
            self.configure_wav2vec(device="cpu")
            audio_time = time.perf_counter() - audio_start
            print(f"[MODELS] ✓ Wav2Vec model loaded in {format_duration(audio_time)}")
        
        # Text Encoder Loading
        print("[MODELS] Loading T5 text encoder...")
        text_start = time.perf_counter()
        self.configure_text_model(device="cpu")
        text_time = time.perf_counter() - text_start
        print(f"[MODELS] ✓ T5 text encoder loaded in {format_duration(text_time)}")
        
        # FSDP Configuration
        print("[MODELS] Configuring FSDP (multi-GPU distribution)...")
        fsdp_start = time.perf_counter()
        # Initialize fsdp.
        self.configure_dit_fsdp_model()
        self.configure_text_fsdp_model()
        fsdp_time = time.perf_counter() - fsdp_start
        print(f"[MODELS] ✓ FSDP configured in {format_duration(fsdp_time)}")
        
        # Final summary
        total_time = time.perf_counter() - start_time
        # GPU status shown above
        
        print(f"[MODELS] 🎯 All models ready in {format_duration(total_time)}")
        print(f"[MODELS]    DiT: {format_duration(dit_time)} | VAE: {format_duration(vae_time)}")
        print(f"[MODELS]    Text: {format_duration(text_time)} | FSDP: {format_duration(fsdp_time)}")
        if audio_time > 0:
            print(f"[MODELS]    Audio: {format_duration(audio_time)}")
        print("")
    
    def configure_dit_model(self, device=get_device()):

        init_unified_parallel(self.config.dit.sp_size)
        self.sp_size = get_unified_parallel_world_size()
        
        # Create dit model.
        init_device = "meta"
        with torch.device(init_device):
            self.dit = create_object(self.config.dit.model)
        self.logger.info(f"Load DiT model on {init_device}.")
        self.dit.eval().requires_grad_(False)

        # Load dit checkpoint.
        path = self.config.dit.checkpoint_dir
        if path.endswith(".pth"):
            state = torch.load(path, map_location=device, mmap=True)
            missing_keys, unexpected_keys = self.dit.load_state_dict(state, strict=False, assign=True)
            self.logger.info(
                f"dit loaded from {path}. "
                f"Missing keys: {len(missing_keys)}, "
                f"Unexpected keys: {len(unexpected_keys)}"
            )
        else:
            from safetensors.torch import load_file
            import json
            def load_custom_sharded_weights(model_dir, base_name, device=device):
                index_path = f"{model_dir}/{base_name}.safetensors.index.json"
                with open(index_path, "r") as f:
                    index = json.load(f)
                weight_map = index["weight_map"]
                shard_files = set(weight_map.values())
                state_dict = {}
                for shard_file in shard_files:
                    shard_path = f"{model_dir}/{shard_file}"
                    shard_state = load_file(shard_path)
                    shard_state = {k: v.to(device) for k, v in shard_state.items()}
                    state_dict.update(shard_state)
                return state_dict
            state = load_custom_sharded_weights(path, 'humo', device)
            self.dit.load_state_dict(state, strict=False, assign=True)
        
        self.dit = meta_non_persistent_buffer_init_fn(self.dit)
        if device in [get_device(), "cuda"]:
            self.dit.to(get_device())

        # Print model size.
        params = sum(p.numel() for p in self.dit.parameters())
        self.logger.info(
            f"[RANK:{get_global_rank()}] DiT Parameters: {clever_format(params, '%.3f')}"
        )
    
    def configure_vae_model(self, device=get_device()):
        self.vae_stride = self.config.vae.vae_stride
        self.vae = WanVAE(
            vae_pth=self.config.vae.checkpoint,
            device=device)
        
        # Simple VAE setup - always on primary GPU
        # Multi-GPU distribution handled at batch level if needed
        num_gpus = getattr(self.config, 'num_gpus', 1)
        if num_gpus > 1:
            print(f"[VAE] Multi-GPU mode: VAE on primary GPU")
        
        if self.config.generation.height == 480:
            self.zero_vae = torch.load(self.config.dit.zero_vae_path)
        elif self.config.generation.height == 720:
            self.zero_vae = torch.load(self.config.dit.zero_vae_720p_path)
        else:
            raise ValueError(f"Unsupported height {self.config.generation.height} for zero-vae.")
    
    def configure_wav2vec(self, device=get_device()):
        audio_separator_model_file = self.config.audio.vocal_separator
        wav2vec_model_path = self.config.audio.wav2vec_model

        self.audio_processor = AudioProcessor(
            16000,
            25,
            wav2vec_model_path,
            "all",
            audio_separator_model_file,
            None,  # not seperate
            os.path.join(self.config.generation.output.dir, "vocals"),
            device=device,
        )

    def configure_text_model(self, device=get_device()):
        self.text_encoder = T5EncoderModel(
            text_len=self.config.dit.model.text_len,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=self.config.text.t5_checkpoint,
            tokenizer_path=self.config.text.t5_tokenizer,
            )

    
    def _setup_multi_gpu_distribution(self, num_gpus):
        """Simple multi-GPU distribution for Replicate production.
        
        Strategy: Split transformer blocks across GPUs evenly.
        This works well for 8xH200 systems.
        """
        total_blocks = len(self.dit.blocks)
        blocks_per_gpu = total_blocks // num_gpus
        extra_blocks = total_blocks % num_gpus
        
        print(f"[DiT] Distributing {total_blocks} blocks across {num_gpus} GPUs")
        
        # Keep embeddings on GPU 0
        self.dit.patch_embedding = self.dit.patch_embedding.cuda(0)
        self.dit.text_embedding = self.dit.text_embedding.cuda(0)
        self.dit.time_embedding = self.dit.time_embedding.cuda(0)
        self.dit.time_projection = self.dit.time_projection.cuda(0)
        
        # Distribute blocks
        block_idx = 0
        for gpu_id in range(num_gpus):
            # Calculate how many blocks this GPU gets
            gpu_blocks = blocks_per_gpu + (1 if gpu_id < extra_blocks else 0)
            
            # Assign blocks to this GPU
            for _ in range(gpu_blocks):
                if block_idx < total_blocks:
                    self.dit.blocks[block_idx] = self.dit.blocks[block_idx].cuda(gpu_id)
                    block_idx += 1
            
            print(f"   GPU {gpu_id}: {gpu_blocks} blocks")
        
        # Head on last GPU
        last_gpu = num_gpus - 1
        self.dit.head = self.dit.head.cuda(last_gpu)
        
        # Audio projection on GPU 0 if present
        if hasattr(self.dit, 'audio_proj'):
            self.dit.audio_proj = self.dit.audio_proj.cuda(0)
        
        # Mark as multi-GPU for forward pass handling
        self.dit._multi_gpu = True
        self.dit._num_gpus = num_gpus
        print(f"[DiT] Multi-GPU distribution complete")
    
    def configure_dit_fsdp_model(self):
        """Configure DiT model for single or multi-GPU inference."""
        from humo.models.wan_modules.model_humo import WanAttentionBlock

        dit_blocks = (WanAttentionBlock,)
        
        # Get number of GPUs
        num_gpus = getattr(self.config, 'num_gpus', 1)
        
        # Simple approach: place model on first GPU
        # For multi-GPU, we'll handle distribution at inference time
        self.dit.to(get_device())
        
        if num_gpus > 1:
            print(f"[DiT] Multi-GPU mode: {num_gpus} GPUs available")
            # For multiple GPUs, use model parallelism
            # Split the 40 transformer blocks across GPUs
            self._setup_multi_gpu_distribution(num_gpus)
        else:
            print(f"[DiT] Single GPU mode")
            
        # No FSDP in single-process Cog environment
        return

        # Init model_shard_cpu_group for saving checkpoint with sharded state_dict.
        init_model_shard_cpu_group(
            self.config.dit.fsdp.sharding_strategy,
            self.config.dit.fsdp.get("device_mesh", None),
        )

        # Assert that dit has wrappable blocks.
        assert any(isinstance(m, dit_blocks) for m in self.dit.modules())

        # Define wrap policy on all dit blocks.
        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            return recurse or isinstance(module, dit_blocks)

        # Configure FSDP settings.
        device_mesh, fsdp_strategy = self.get_fsdp_sharding_config(
            self.config.dit.fsdp.sharding_strategy,
            self.config.dit.fsdp.get("device_mesh", None),
        )
        settings = dict(
            auto_wrap_policy=custom_auto_wrap_policy,
            sharding_strategy=fsdp_strategy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=get_local_rank(),
            use_orig_params=False,
            sync_module_states=True,
            forward_prefetch=True,
            limit_all_gathers=False,  # False for ZERO2.
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,  # H200 has fast BF16 reduction
                buffer_dtype=torch.bfloat16,  # H200 handles BF16 buffers efficiently
            ),
            device_mesh=device_mesh,
            param_init_fn=meta_param_init_fn,
        )

        # Apply FSDP.
        self.dit = FullyShardedDataParallel(self.dit, **settings)
        
        # Optional: Compile DiT model for H200 performance (PyTorch 2.0+)
        # Disable by setting HUMO_COMPILE=0 if issues arise
        if os.environ.get("HUMO_COMPILE", "1") == "1" and hasattr(torch, 'compile'):
            try:
                print("[DiT] Compiling model with torch.compile (mode=reduce-overhead)")
                self.dit = torch.compile(self.dit, mode="reduce-overhead", fullgraph=False)
            except Exception as e:
                print(f"[DiT] torch.compile failed (continuing without): {e}")
        # self.dit.to(get_device())


    def configure_text_fsdp_model(self):
        # Simple setup - no FSDP for single-process Cog
        if not self.config.text.fsdp.enabled:
            self.text_encoder.to(get_device())
            num_gpus = getattr(self.config, 'num_gpus', 1)
            if num_gpus > 1:
                print(f"[Text] Multi-GPU mode: text encoder on primary GPU")
            return

        # from transformers.models.t5.modeling_t5 import T5Block
        from humo.models.wan_modules.t5 import T5SelfAttention

        text_blocks = (torch.nn.Embedding, T5SelfAttention)
        # text_blocks_names = ("QWenBlock", "QWenModel")  # QWen cannot be imported. Use str.

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            return (
                recurse
                or isinstance(module, text_blocks)
            )

        # Apply FSDP.
        text_encoder_dtype = getattr(torch, self.config.text.dtype)
        device_mesh, fsdp_strategy = self.get_fsdp_sharding_config(
            self.config.text.fsdp.sharding_strategy,
            self.config.text.fsdp.get("device_mesh", None),
        )
        self.text_encoder = FullyShardedDataParallel(
            module=self.text_encoder,
            auto_wrap_policy=custom_auto_wrap_policy,
            sharding_strategy=fsdp_strategy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=get_local_rank(),
            use_orig_params=False,
            sync_module_states=False,
            forward_prefetch=True,
            limit_all_gathers=True,
            mixed_precision=MixedPrecision(
                param_dtype=text_encoder_dtype,
                reduce_dtype=text_encoder_dtype,
                buffer_dtype=text_encoder_dtype,
            ),
            device_mesh=device_mesh,
        )
        self.text_encoder.to(get_device()).requires_grad_(False)


    def load_image_latent_ref_id(self, path: str, size, device):
        # Load size.
        h, w = size[1], size[0]

        # Load image.
        if len(path) > 1 and not isinstance(path, str):
            ref_vae_latents = []
            for image_path in path:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")

                    # Calculate the required size to keep aspect ratio and fill the rest with padding.
                    img_ratio = img.width / img.height
                    target_ratio = w / h
                    
                    if img_ratio > target_ratio:  # Image is wider than target
                        new_width = w
                        new_height = int(new_width / img_ratio)
                    else:  # Image is taller than target
                        new_height = h
                        new_width = int(new_height * img_ratio)
                    
                    # img = img.resize((new_width, new_height), Image.ANTIALIAS)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                    # Create a new image with the target size and place the resized image in the center
                    delta_w = w - img.size[0]
                    delta_h = h - img.size[1]
                    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                    new_img = ImageOps.expand(img, padding, fill=(255, 255, 255))

                    # Transform to tensor and normalize.
                    transform = Compose(
                        [
                            ToTensor(),
                            Normalize(0.5, 0.5),
                        ]
                    )
                    new_img = transform(new_img)
                    # img_vae_latent = self.vae_encode([new_img.unsqueeze(1)])[0]
                    img_vae_latent = self.vae.encode([new_img.unsqueeze(1)], device)
                    ref_vae_latents.append(img_vae_latent[0])

            return [torch.cat(ref_vae_latents, dim=1)]
        else:
            if not isinstance(path, str):
                path = path[0]
            with Image.open(path) as img:
                img = img.convert("RGB")

                # Calculate the required size to keep aspect ratio and fill the rest with padding.
                img_ratio = img.width / img.height
                target_ratio = w / h
                
                if img_ratio > target_ratio:  # Image is wider than target
                    new_width = w
                    new_height = int(new_width / img_ratio)
                else:  # Image is taller than target
                    new_height = h
                    new_width = int(new_height * img_ratio)
                
                # img = img.resize((new_width, new_height), Image.ANTIALIAS)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Create a new image with the target size and place the resized image in the center
                delta_w = w - img.size[0]
                delta_h = h - img.size[1]
                padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                new_img = ImageOps.expand(img, padding, fill=(255, 255, 255))

                # Transform to tensor and normalize.
                transform = Compose(
                    [
                        ToTensor(),
                        Normalize(0.5, 0.5),
                    ]
                )
                new_img = transform(new_img)
                img_vae_latent = self.vae.encode([new_img.unsqueeze(1)], device)

            # Vae encode.
            return img_vae_latent
    
    def get_audio_emb_window(self, audio_emb, frame_num, frame0_idx, audio_shift=2):
        zero_audio_embed = torch.zeros((audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)
        zero_audio_embed_3 = torch.zeros((3, audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)  # device=audio_emb.device
        iter_ = 1 + (frame_num - 1) // 4
        audio_emb_wind = []
        for lt_i in range(iter_):
            if lt_i == 0:
                st = frame0_idx + lt_i - 2
                ed = frame0_idx + lt_i + 3
                wind_feat = torch.stack([
                    audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                    for i in range(st, ed)
                ], dim=0)
                wind_feat = torch.cat((zero_audio_embed_3, wind_feat), dim=0)
            else:
                st = frame0_idx + 1 + 4 * (lt_i - 1) - audio_shift
                ed = frame0_idx + 1 + 4 * lt_i + audio_shift
                wind_feat = torch.stack([
                    audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                    for i in range(st, ed)
                ], dim=0)
            audio_emb_wind.append(wind_feat)
        audio_emb_wind = torch.stack(audio_emb_wind, dim=0)

        return audio_emb_wind, ed - audio_shift
    
    def audio_emb_enc(self, audio_emb, wav_enc_type="whisper"):
        if wav_enc_type == "wav2vec":
            feat_merge = audio_emb
        elif wav_enc_type == "whisper":
            feat0 = linear_interpolation_fps(audio_emb[:, :, 0: 8].mean(dim=2), 50, 25)
            feat1 = linear_interpolation_fps(audio_emb[:, :, 8: 16].mean(dim=2), 50, 25)
            feat2 = linear_interpolation_fps(audio_emb[:, :, 16: 24].mean(dim=2), 50, 25)
            feat3 = linear_interpolation_fps(audio_emb[:, :, 24: 32].mean(dim=2), 50, 25)
            feat4 = linear_interpolation_fps(audio_emb[:, :, 32], 50, 25)
            feat_merge = torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0]
        else:
            raise ValueError(f"Unsupported wav_enc_type: {wav_enc_type}")
        
        return feat_merge
    
    def parse_output(self, output):
        latent = output[0]
        mask = None
        return latent, mask
    
    def forward_tia(self, latents, timestep, t, step_change, arg_tia, arg_ti, arg_i, arg_null, scale_t=None, scale_a=None):
        # Use override scales if provided, otherwise fall back to config
        scale_t = scale_t if scale_t is not None else self.config.generation.scale_t
        scale_a = scale_a if scale_a is not None else self.config.generation.scale_a
        
        pos_tia, _ = self.parse_output(self.dit(
            latents, t=timestep, **arg_tia
            ))
        self.maybe_empty_cache()

        pos_ti, _ = self.parse_output(self.dit(
            latents, t=timestep, **arg_ti
            ))
        self.maybe_empty_cache()

        if t > step_change:
            neg, _ = self.parse_output(self.dit(
                latents, t=timestep, **arg_i
                ))  # img included in null, same with official Wan-2.1
            self.maybe_empty_cache()

            noise_pred = scale_a * (pos_tia - pos_ti) + \
                    scale_t * (pos_ti - neg) + \
                    neg
        else:
            neg, _ = self.parse_output(self.dit(
                latents, t=timestep, **arg_null
                ))  # img not included in null
            self.maybe_empty_cache()

            noise_pred = scale_a * (pos_tia - pos_ti) + \
                    (scale_t - 2.0) * (pos_ti - neg) + \
                    neg
        return noise_pred
    
    def forward_ta(self, latents, timestep, arg_ta, arg_t, arg_null, scale_t=None, scale_a=None):
        # Use override scales if provided, otherwise fall back to config
        scale_t = scale_t if scale_t is not None else self.config.generation.scale_t
        scale_a = scale_a if scale_a is not None else self.config.generation.scale_a
        
        pos_ta, _ = self.parse_output(self.dit(
            latents, t=timestep, **arg_ta
            ))
        self.maybe_empty_cache()

        pos_t, _ = self.parse_output(self.dit(
            latents, t=timestep, **arg_t
            ))
        self.maybe_empty_cache()

        neg, _ = self.parse_output(self.dit(
                latents, t=timestep, **arg_null
                ))
        self.maybe_empty_cache()
            
        noise_pred = scale_a * (pos_ta - pos_t) + \
                scale_t * (pos_t - neg) + \
                neg
        return noise_pred
                    
    @torch.no_grad()
    def inference(self,
                 input_prompt,
                 img_path,
                 audio_path,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 mode_override=None,
                 scale_t_override=None,
                 scale_a_override=None,
                 device = get_device(),
        ):
        """Core inference method: text + optional image/audio → video tensor.
        
        Args:
            mode_override: Force generation mode ('TA' or 'TIA'). If None, uses config.
            offload_model: Move models to CPU between stages to save VRAM.
        
        Returns:
            torch.Tensor: Video in [C, F, H, W] format (caller must transpose for writing).
        """

        # Start total inference timing
        inference_start_time = time.perf_counter()
        print(f"[INFERENCE] Starting inference for prompt: \"{input_prompt[:50]}...\"" if len(input_prompt) > 50 else f"[INFERENCE] Starting inference for prompt: \"{input_prompt}\"")
        log_multi_gpu_utilization("INFERENCE-START")

        # Ensure VAE is on the right device
        self.vae.model.to(device=device)
        if img_path is not None:
            latents_ref = self.load_image_latent_ref_id(img_path, size, device)
        else:
            latents_ref = [torch.zeros(16, 1, size[1]//8, size[0]//8).to(device)]
            
        if offload_model:
            self.vae.model.to(device="cpu")
        latents_ref_neg = [torch.zeros_like(latent_ref) for latent_ref in latents_ref]
        
        # audio
        if audio_path is not None:
            if self.config.generation.extract_audio_feat:
                # Whisper on main device
                self.audio_processor.whisper.to(device=device)
                audio_emb, audio_length = self.audio_processor.preprocess(audio_path)
                if offload_model:
                    self.audio_processor.whisper.to(device='cpu')
            else:
                audio_emb_path = audio_path.replace(".wav", ".pt")
                audio_emb = torch.load(audio_emb_path).to(device=device)
                audio_emb = self.audio_emb_enc(audio_emb, wav_enc_type="whisper")
                self.logger.info("使用预先提取好的音频特征: %s", audio_emb_path)
        else:
            audio_emb = torch.zeros(frame_num, 5, 1280).to(device)
            
        frame_num = frame_num if frame_num != -1 else audio_length
        frame_num = 4 * ((frame_num - 1) // 4) + 1
        audio_emb, _ = self.get_audio_emb_window(audio_emb, frame_num, frame0_idx=0)
        zero_audio_pad = torch.zeros(latents_ref[0].shape[1], *audio_emb.shape[1:]).to(audio_emb.device)
        audio_emb = torch.cat([audio_emb, zero_audio_pad], dim=0)
        audio_emb = [audio_emb.to(device)]
        audio_emb_neg = [torch.zeros_like(audio_emb[0])]
        
        # preprocess
        self.patch_size = self.config.dit.model.patch_size
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1 + latents_ref[0].shape[1],
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.config.generation.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)

        # Simple text encoder handling
        if hasattr(self.text_encoder, 'model'):
            self.text_encoder.model.to(device)
        
        context = self.text_encoder([input_prompt], device)
        context_null = self.text_encoder([n_prompt], device)
        
        if offload_model and hasattr(self.text_encoder, 'model'):
            self.text_encoder.model.cpu()

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1], # - latents_ref[0].shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.dit, 'no_sync', noop_no_sync)
        step_change = self.config.generation.step_change # 980

        # evaluation mode
        with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.inference_mode(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=1000,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=device, shift=shift)
                timesteps = sample_scheduler.timesteps

            # sample videos
            latents = noise

            msk = torch.ones(4, target_shape[1], target_shape[2], target_shape[3], device=get_device())
            msk[:,:-latents_ref[0].shape[1]] = 0

            zero_vae = self.zero_vae[:, :(target_shape[1]-latents_ref[0].shape[1])].to(
                device=get_device(), dtype=latents_ref[0].dtype)
            y_c = torch.cat([
                zero_vae,
                latents_ref[0]
                ], dim=1)
            y_c = [torch.concat([msk, y_c])]

            y_null = self.zero_vae[:, :target_shape[1]].to(
                device=get_device(), dtype=latents_ref[0].dtype)
            y_null = [torch.concat([msk, y_null])]

            arg_null = {'seq_len': seq_len, 'audio': audio_emb_neg, 'y': y_null, 'context': context_null}
            arg_t = {'seq_len': seq_len, 'audio': audio_emb_neg, 'y': y_null, 'context': context}
            arg_i = {'seq_len': seq_len, 'audio': audio_emb_neg, 'y': y_c, 'context': context_null}
            arg_ti = {'seq_len': seq_len, 'audio': audio_emb_neg, 'y': y_c, 'context': context}
            arg_ta = {'seq_len': seq_len, 'audio': audio_emb, 'y': y_null, 'context': context}
            arg_tia = {'seq_len': seq_len, 'audio': audio_emb, 'y': y_c, 'context': context}
            
            self.maybe_empty_cache()
            # Handle device placement
            if not hasattr(self.dit, '_multi_gpu') or not self.dit._multi_gpu:
                # Single GPU mode
                self.dit.to(device=get_device())
            # For multi-GPU, model is already distributed
            # Clean sampling with essential progress tracking
            total_steps = len(timesteps)
            step_times = []
            mode_to_use = mode_override if mode_override is not None else self.config.generation.mode
            num_gpus = getattr(self.config, "num_gpus", 1)
            
            print(f"[INFERENCE] Starting {mode_to_use} sampling: {total_steps} steps across {num_gpus} GPU(s)")
            
            # Show model distribution status for multi-GPU
            if num_gpus > 1:
                log_model_distribution_status(self.dit, "DiT")
            
            # Initial GPU status logged in predict.py
            
            # Simple progress with periodic updates (every 20% or 10 steps, whichever is larger)
            update_interval = max(total_steps // 5, 10)
            
            for step_idx, t in enumerate(timesteps):
                step_start = time.perf_counter()
                timestep = t.unsqueeze(0)

                if mode_to_use == "TIA":
                    noise_pred = self.forward_tia(latents, timestep, t, step_change, 
                                                  arg_tia, arg_ti, arg_i, arg_null,
                                                  scale_t=scale_t_override, scale_a=scale_a_override)
                elif mode_to_use == "TA":
                    noise_pred = self.forward_ta(latents, timestep, arg_ta, arg_t, arg_null,
                                                scale_t=scale_t_override, scale_a=scale_a_override)
                else:
                    raise ValueError(f"Unsupported generation mode: {mode_to_use}")

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

                del timestep
                self.maybe_empty_cache()
                
                # Track timing
                step_time = time.perf_counter() - step_start
                step_times.append(step_time)
                
                # Progress updates (not every step to avoid spam)
                if (step_idx + 1) % update_interval == 0 or step_idx == 0 or step_idx == total_steps - 1:
                    avg_time = sum(step_times) / len(step_times)
                    remaining = total_steps - step_idx - 1
                    eta = format_time_remaining(avg_time, step_idx + 1, total_steps)
                    progress = ((step_idx + 1) / total_steps) * 100
                    
                    print(f"[INFERENCE] Step {step_idx + 1}/{total_steps} ({progress:.0f}%) | "
                          f"Avg: {avg_time:.1f}s/step | ETA: {eta}")
                    
                    # GPU check at major milestones for multi-GPU systems
                    if num_gpus > 1 and (step_idx + 1) % (total_steps // 2) == 0:
                        log_multi_gpu_utilization("MID")
            
            # Sampling complete
            total_sampling_time = sum(step_times)
            print(f"[INFERENCE] ✓ Sampling complete: {total_steps} steps in {format_duration(total_sampling_time)}")
            
            # Final GPU utilization
            # Final GPU status logged in predict.py
            
            x0 = latents
            x0 = [x0_[:,:-latents_ref[0].shape[1]] for x0_ in x0]

            if offload_model:
                self.dit.cpu()
            self.maybe_empty_cache()
            # if get_local_rank() == 0:
            # VAE Decoding with progress tracking
            print("[INFERENCE] Starting VAE decode...")
            log_multi_gpu_utilization("VAE-DECODE")
            decode_start = time.perf_counter()
            
            # Ensure VAE is on device for decoding
            self.vae.model.to(device=device)
            videos = self.vae.decode(x0)
            
            decode_time = time.perf_counter() - decode_start
            print(f"[INFERENCE] ✓ VAE decode complete in {format_duration(decode_time)}")
            log_multi_gpu_utilization("VAE-COMPLETE")
            
            if offload_model:
                self.vae.model.to(device="cpu")

        del noise, latents, noise_pred
        del audio_emb, audio_emb_neg, latents_ref, latents_ref_neg, context, context_null
        del x0, temp_x0
        del sample_scheduler
        self.maybe_empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()
        # Final performance summary
        total_inference_time = time.perf_counter() - inference_start_time
        log_inference_performance_summary(step_times, total_sampling_time, num_gpus)
        print(f"[INFERENCE] 🎯 Total inference time: {format_duration(total_inference_time)}")

        return videos[0] # if get_local_rank() == 0 else None


    def inference_loop(self):
        gen_config = self.config.generation
        pos_prompts = self.prepare_positive_prompts()
        
        # Create output dir.
        os.makedirs(gen_config.output.dir, exist_ok=True)

        # Start generation.
        for prompt in pos_prompts:
            seed = self.config.generation.seed
            seed = seed if seed is not None else random.randint(0, 100000)

            audio_path = prompt.get("audio", None)
            ref_img_path = prompt.get("ref_img", None)
            itemname = prompt.get("itemname", None)
            if "I" not in self.config.generation.mode:
                ref_img_path = None
            if "A" not in self.config.generation.mode:
                audio_path = None

            video = self.inference(
                prompt.text,
                ref_img_path,
                audio_path,
                size=SIZE_CONFIGS[f"{gen_config.width}*{gen_config.height}"],
                frame_num=gen_config.frames,
                shift=self.config.diffusion.timesteps.sampling.shift,
                sample_solver='unipc',
                sampling_steps=self.config.diffusion.timesteps.sampling.steps,
                seed=seed,
                offload_model=False,
            )

            torch.cuda.empty_cache()
            gc.collect()
            

            # Save samples.
            if get_sequence_parallel_rank() == 0:
                pathname = self.save_sample(
                    sample=video,
                    audio_path=audio_path,
                    itemname=itemname,
                )
                self.logger.info(f"Finished {itemname}, saved to {pathname}.")
            
            del video, prompt
            torch.cuda.empty_cache()
            gc.collect()
            

    def save_sample(self, *, sample: torch.Tensor, audio_path: str, itemname: str):
        gen_config = self.config.generation
        # Prepare file path.
        extension = ".mp4" if sample.ndim == 4 else ".png"
        filename = f"{itemname}_seed{gen_config.seed}"
        filename += extension
        pathname = os.path.join(gen_config.output.dir, filename)
        # Convert sample.
        sample = sample.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).to("cpu", torch.uint8)
        sample = rearrange(sample, "c t h w -> t h w c")
        # Save file.
        if sample.ndim == 4:
            if audio_path is not None:
                tensor_to_video(
                    sample.numpy(),
                    pathname,
                    audio_path,
                    fps=gen_config.fps)
            else:
                mediapy.write_video(
                path=pathname,
                images=sample.numpy(),
                fps=gen_config.fps,
            )
        else:
            raise ValueError
        return pathname
    

    def prepare_positive_prompts(self):
        pos_prompts = self.config.generation.positive_prompt
        if pos_prompts.endswith(".json"):
            pos_prompts = prepare_json_dataset(pos_prompts)
        else:
            raise NotImplementedError
        assert isinstance(pos_prompts, ListConfig)

        return pos_prompts