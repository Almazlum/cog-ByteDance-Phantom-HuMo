import os
import tempfile
import torch
import random
import subprocess
import numpy as np
import time
from typing import Optional
from tqdm import tqdm

from cog import BasePredictor, Input, Path as CPath
from pathlib import Path
from omegaconf import OmegaConf
from humo.generate import Generator
from common.config import load_config


# Weight download configuration
WEIGHTS_DIR = "weights"  # Relative to /src in container
WEIGHTS_BASE_URL = "https://weights.replicate.delivery/default/HuMo/weights/"

# Set environment variables for model caching - needs to happen early
os.environ["HF_HOME"] = WEIGHTS_DIR
os.environ["TORCH_HOME"] = WEIGHTS_DIR  
os.environ["HF_DATASETS_CACHE"] = WEIGHTS_DIR
os.environ["TRANSFORMERS_CACHE"] = WEIGHTS_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = WEIGHTS_DIR
# Recommended for Hopper to improve kernel scheduling
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

def download_weights_fast(url: str, dest: str) -> None:
    """Download model weights using pget with parallel downloads"""
    start = time.time()
    print(f"[!] Downloading from: {url}")
    print(f"[~] Destination: {dest}")
    
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    
    try:
        print(f"[~] Running: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
        print(f"[✓] Download completed in {time.time() - start:.1f}s")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Download failed: {e}")
        raise

def download_weights_fallback(repo: str, dest: str, desc: str) -> None:
    """Fallback to Hugging Face download if CDN fails"""
    print(f"📦 Fallback: Downloading {desc} from Hugging Face...")
    os.makedirs(dest, exist_ok=True)
    subprocess.run([
        "huggingface-cli", "download", repo, "--local-dir", dest, "--quiet"
    ], check=True, timeout=3600)
    print(f"✅ {desc} ready via Hugging Face")

def download_weights():
    """Download all required model weights for HuMo-17B.
    
    Uses fast CDN download via pget, falls back to Hugging Face if CDN fails.
    Weights are cached in WEIGHTS_DIR to avoid re-downloading on container restart.
    """
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    # Model weight tar files uploaded to CDN
    model_files = [
        "HuMo.tar",
        "Wan2.1-T2V-1.3B.tar", 
        "whisper-large-v3.tar",
        "audio_separator.tar"
    ]
    
    # HF fallback repositories
    fallback_repos = {
        "HuMo.tar": ("bytedance-research/HuMo", "HuMo-17B model weights"),
        "Wan2.1-T2V-1.3B.tar": ("Wan-AI/Wan2.1-T2V-1.3B", "VAE and text encoder"),
        "whisper-large-v3.tar": ("openai/whisper-large-v3", "Audio encoder"),
        "audio_separator.tar": ("huangjackson/Kim_Vocal_2", "Audio separator")
    }
    
    for model_file in model_files:
        url = WEIGHTS_BASE_URL + model_file
        filename = url.split("/")[-1]
        dest_path = os.path.join(WEIGHTS_DIR, filename)
        extracted_dir = dest_path.replace(".tar", "")
        
        # Check if already extracted
        if not os.path.exists(extracted_dir):
            try:
                # Try fast CDN download first
                hf_repo, desc = fallback_repos[model_file]
                print(f"🚀 Fast download: {desc}...")
                download_weights_fast(url, dest_path)
                print(f"✅ {desc} ready via CDN")
            except Exception as e:
                print(f"⚠️  CDN download failed: {e}")
                # Fallback to Hugging Face
                hf_repo, desc = fallback_repos[model_file]
                download_weights_fallback(hf_repo, extracted_dir, desc)
        else:
            _, desc = fallback_repos[model_file]
            print(f"✅ {desc} ready")


def setup_gpu_environment():
    """Detect GPU count and configure environment for Replicate production.
    
    Simple approach:
    - Single GPU: Normal operation
    - Multiple GPUs: Model parallelism via device placement
    
    Returns:
        tuple: (num_gpus, sp_size) for config selection
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPUs detected. HuMo requires GPU acceleration.")
        
    num_gpus = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    
    print(f"🔍 Detected {num_gpus} GPU(s): {gpu_names}")
    
    # Always use all available GPUs
    gpu_ids = ','.join(str(i) for i in range(num_gpus))
    
    if self.num_gpus > 1:
        print(f"🚀 Multi-GPU: {num_gpus} H200s detected")
        print(f"   All GPUs will be utilized for maximum performance")
        # For Replicate production with 8xH200s
        print(f"   GPUs: {gpu_ids}")
    else:
        print(f"⚡ Single GPU: {gpu_names[0]}")
    
    # Simple environment setup - no distributed complexity
    os.environ.update({
        'CUDA_VISIBLE_DEVICES': gpu_ids,
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '12355',
        'RANK': '0',
        'WORLD_SIZE': '1',  # Always single process for Cog/Replicate
        'LOCAL_RANK': '0',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'
    })
    
    return num_gpus, 1  # sp_size=1 for single process


class Predictor(BasePredictor):
    """HuMo-17B Predictor optimized for Replicate production.
    
    Automatically detects and uses all available H200 GPUs.
    On 8xH200 systems, the model will utilize all GPUs for maximum performance.
    """
    def setup(self) -> None:
        """One-time model setup.

        - Detects GPUs and configures distributed/env variables
        - Downloads weights (CDN fast-path, HF fallback)
        - Loads single- vs multi-GPU config, builds generator
        - Enables fast matmul/attention backends on Hopper
        - Performs a tiny warmup so first predict() is fast
        """
        print("🚀 Setting up HuMo-17B...")
        
        # Configure GPU environment
        num_gpus, sp_size = setup_gpu_environment()
        self.num_gpus = num_gpus  # Store for use in predict method
        torch.backends.cudnn.benchmark = True
        # H200-specific optimizations
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        
        # Enable Hopper/H200 optimizations
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(False)
        
        # H200 specific: Enable FP8 for attention if available (requires PyTorch 2.4+)
        if hasattr(torch.backends.cuda, 'enable_fp8_attention'):
            torch.backends.cuda.enable_fp8_attention(True)
            
        # Use medium precision for better H200 performance (allows TensorCore usage)
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision("medium")
            
        # H200: Enable TMA (Tensor Memory Accelerator) if available
        if hasattr(torch.cuda, 'set_tma_enabled'):
            torch.cuda.set_tma_enabled(True)
        # Silence tokenizers fork warning to avoid overhead
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        
        # Download weights
        download_weights()
        
        # Always use single GPU config for Replicate/Cog
        config = load_config("humo/configs/inference/generate_single_gpu.yaml")
        
        # Store GPU count for the generator
        config.num_gpus = num_gpus
        
        if self.num_gpus > 1:
            print(f"📊 Multi-GPU mode: {num_gpus} H200s will be utilized")
            print(f"   Model parallelism will be handled automatically")
        else:
            print("📊 Single GPU mode")
        
        # Initialize generator
        self.generator = Generator(config)
        self.generator.configure_models()
        
        print(f"✅ HuMo-17B ready on {num_gpus} GPU(s)")

        # Optional warmup to prime kernels and caches (setup can take longer; predict should be fast)
        try:
            print("🔥 Warming up kernels (2 steps, 1 frame)...")
            _ = self.generator.inference(
                input_prompt="warmup",
                img_path=None,
                audio_path=None,
                size=(1280, 720),
                frame_num=1,
                sampling_steps=2,
                shift=1.0,
                n_prompt="",
                seed=123,
                offload_model=False,
            )
            print("🔥 Warmup complete")
        except Exception as e:
            print(f"⚠️  Warmup skipped: {e}")

    def predict(
        self,
        prompt: str = Input(
            description="Text description of the video. Be detailed about the person, actions, and scene.",
            default="A person walking confidently down a busy street"
        ),
        reference_image: Optional[CPath] = Input(
            description="Reference image to control the person's appearance (optional)",
            default=None,
        ),
        audio: Optional[CPath] = Input(
            description="Audio file for lip-sync and movement synchronization (optional)",
            default=None,
        ),
        width: int = Input(
            description="Video width in pixels",
            default=1280,
            choices=[1280]  # 720p only for now (480p has zero_vae compatibility issues)
        ),
        height: int = Input(
            description="Video height in pixels", 
            default=720,
            choices=[720]  # 720p only for now (480p has zero_vae compatibility issues)
        ),
        num_frames: int = Input(
            description="Number of frames (25 fps, so 25 frames = 1 second). Model trained on up to 97 frames.",
            default=49,
            ge=9,  # Minimum for meaningful motion (increased from 1)
            le=97  # Model's training limit
        ),
        num_inference_steps: int = Input(
            description="Denoising steps. More steps = higher quality but slower. Research default is 50.",
            default=50,  # Match research default for quality
            ge=10,  # Minimum for decent quality (increased from 5)
            le=100
        ),
        guidance_scale: float = Input(
            description="Text guidance strength. Research default is 5.0. Lower values (3-5) often produce more natural lighting.",
            default=5.0,  # Match research default
            ge=2.0,  # Minimum for meaningful guidance (increased from 1.0)
            le=15.0  # Reduced max to prevent over-guidance
        ),
        audio_guidance_scale: float = Input(
            description="Audio guidance strength (when audio provided). Higher = better sync. Research default is 5.5.",
            default=5.5,  # Match research default
            ge=2.0,  # Minimum for meaningful guidance
            le=15.0  # Reduced max to prevent over-guidance
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducible generation",
            default=None,
            ge=0,
            le=2147483647
        ),
        negative_prompt: str = Input(
            description="What to avoid in the video",
            default="blurry, low quality, distorted, bad anatomy"  # Simplified, avoid anti-lighting terms
        ),
    ) -> CPath:
        """Generate a video from text (+ optional image/audio).

        Defaults target the model's trained resolution and frame budget.
        API users can override frames/steps/guidance while keeping sensible behavior.
        """
        # Validate inputs
        if not prompt.strip() or len(prompt.strip()) < 10:
            raise ValueError("Prompt must be at least 10 characters")
        
        # Resolution validation (720p only for stable operation)
        if width != 1280 or height != 720:
            raise ValueError(f"Invalid resolution {width}x{height}. Only 1280x720 (720p) is currently supported.")
        
        # Set random seed
        if seed is None:
            seed = random.randint(0, 100000)
        
        # Determine mode
        mode = "T"
        if reference_image:
            mode += "I"
        if audio:
            mode += "A"

        # Map user-facing mode to generator modes (TA or TIA only)
        if audio:
            generator_mode = "TIA" if reference_image else "TA"
        else:
            generator_mode = "TA"  # Text-only runs via TA path with zeroed audio
        
        # Log generation details
        duration = num_frames / 25.0
        print(f"🎬 Generating {duration:.1f}s video ({width}x{height}, {num_frames} frames)")
        print(f"📝 Mode: {mode} (engine={generator_mode}) | Steps: {num_inference_steps} | Seed: {seed}")
        print(f"🖥️  Hardware: {self.num_gpus}x H200 GPU(s) | Multi-GPU: {"Enabled" if self.num_gpus > 1 else "Single GPU"}")

        # Early validation: check file paths before expensive GPU operations
        if reference_image is not None:
            ref_path = str(reference_image)
            if not os.path.isfile(ref_path):
                raise ValueError(f"Reference image not found: {ref_path}")
        if audio is not None:
            aud_path = str(audio)
            if not os.path.isfile(aud_path):
                raise ValueError(f"Audio file not found: {aud_path}")
        
        # Pass guidance scales as parameters (config is read-only)

        # Generate video using the configured model
        # Show initial GPU status for multi-GPU validation
        if self.num_gpus > 1:
            from humo.generate import log_multi_gpu_utilization
            print("\n🔍 Multi-GPU Status Check:")
            log_multi_gpu_utilization("PRE-INFERENCE", show_detailed=False)
        
        try:
            video_tensor = self.generator.inference(
                input_prompt=prompt,
                img_path=str(reference_image) if reference_image else None,
                audio_path=str(audio) if audio else None,
                size=(width, height),
                frame_num=num_frames,
                sampling_steps=num_inference_steps,
                shift=5.0,
                sample_solver='unipc',  # UniPC scheduler for stable generation
                n_prompt=negative_prompt,
                seed=seed,
                offload_model=False,  # Keep models on GPU for speed
                mode_override=generator_mode,
                scale_t_override=float(guidance_scale),
                scale_a_override=float(audio_guidance_scale),
            )
            print("🎥 Generation complete!")
            
            # Final GPU utilization check for multi-GPU systems
            if self.num_gpus > 1:
                print("\n📊 Final Multi-GPU Utilization:")
                log_multi_gpu_utilization("POST-INFERENCE", show_detailed=False)
            
        except Exception as e:
            # Provide actionable error messages for common failures
            if "CUDA out of memory" in str(e):
                raise RuntimeError("GPU memory insufficient. Try smaller resolution or fewer frames.")
            raise RuntimeError(f"Generation failed: {str(e)}")
        
        # Create simple output path
        output_path = "/tmp/video.mp4"
        video_np = video_tensor.cpu().numpy()
        
        if video_np.ndim != 4:
            raise ValueError(f"Invalid video tensor shape: {video_np.shape}")

        # Apply proper normalization like the original research code
        # Original: sample.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).to("cpu", torch.uint8)
        video_tensor_normalized = video_tensor.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).to("cpu", torch.uint8)
        video_np = video_tensor_normalized.numpy()
        
        # Fix tensor layout: model outputs [C, F, H, W] but imageio expects [F, H, W, C]
        video_np = np.transpose(video_np, (1, 2, 3, 0))
        
        if audio:
            from humo.models.utils.utils import tensor_to_video
            # Convert back to float for tensor_to_video (it expects [0, 1] range)
            video_np_float = video_np.astype(np.float32) / 255.0
            tensor_to_video(video_np_float, output_path, str(audio), fps=25)
            print("🎵 Saved with audio sync")
        else:
            import imageio
            with imageio.get_writer(output_path, fps=25, codec='libx264', quality=8) as writer:
                for frame in video_np:
                    # Frame is already properly normalized uint8
                    writer.append_data(frame)
            print("🎬 Saved video")
        
        print(f"✅ Success: {duration:.1f}s video at {width}x{height}")
        
        # Return the tempfile path directly
        return CPath(output_path)