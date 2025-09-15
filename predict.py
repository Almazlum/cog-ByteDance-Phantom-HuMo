"""
HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning
Cog implementation for Replicate deployment
"""

import os
import sys
import gc
import subprocess
import time
import logging
from pathlib import Path
from typing import Iterator, Any, Optional, Union
import tempfile
import json

# GPU optimizations
os.environ.update({
    "CUDA_LAUNCH_BLOCKING": "0",           
    "TORCH_BACKENDS_CUDNN_BENCHMARK": "1", 
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",  
    "OMP_NUM_THREADS": "8",                
    "MKL_NUM_THREADS": "8",               
    "TOKENIZERS_PARALLELISM": "false"      
})

# Cache Manager Integration
MODEL_CACHE = "weights"
BASE_URL = "https://weights.replicate.delivery/default/HuMo/weights/"

# Set environment variables for model caching
os.environ.update({
    "HF_HOME": MODEL_CACHE,
    "TORCH_HOME": MODEL_CACHE,
    "HF_DATASETS_CACHE": MODEL_CACHE,
    "TRANSFORMERS_CACHE": MODEL_CACHE,
    "HUGGINGFACE_HUB_CACHE": MODEL_CACHE
})

def download_weights(url: str, dest: str) -> None:
    """Download model weights using pget with parallel downloads"""
    start = time.time()
    print(f"[!] Initiating download from URL: {url}")
    print(f"[~] Destination path: {dest}")
    
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
        print(f"[✓] Download completed in {time.time() - start:.1f}s")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}.")
        raise

def ensure_model_weights():
    """Download and extract all required model weights"""
    models_to_download = [
        ("HuMo.tar", "HuMo"),
        ("Wan2.1-T2V-1.3B.tar", "Wan2.1-T2V-1.3B"), 
        ("whisper-large-v3.tar", "whisper-large-v3"),
        ("audio_separator.tar", "audio_separator")
    ]
    
    for tar_name, extracted_name in models_to_download:
        target_path = Path(MODEL_CACHE) / extracted_name
        if not target_path.exists():
            print(f"[!] {extracted_name} not found, downloading...")
            download_weights(f"{BASE_URL}{tar_name}", str(target_path))
        else:
            print(f"[✓] {extracted_name} already exists, skipping download")

# Import after environment setup
import torch
import numpy as np
from cog import BasePredictor, Input, Path as CogPath

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("[!] Starting HuMo model setup...")
        
        # Ensure all model weights are available
        ensure_model_weights()
        
        # Add humo module to Python path
        sys.path.insert(0, str(Path.cwd()))
        
        # Import HuMo modules
        try:
            from humo.models.humo_model import HuMoModel
            from humo.utils.config_utils import load_config
            from humo.utils.inference_utils import prepare_inputs, generate_video
            
            print("[✓] HuMo modules imported successfully")
            
            # Load model configuration
            config_path = "humo/configs/inference/generate.yaml"
            self.config = load_config(config_path)
            
            # Initialize model
            print("[!] Loading HuMo 17B model...")
            self.model = HuMoModel.from_pretrained(
                f"{MODEL_CACHE}/HuMo",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            print("[✓] HuMo model loaded successfully")
            
            # Load auxiliary models
            print("[!] Loading auxiliary models...")
            
            # VAE and Text Encoder (Wan2.1)
            from diffusers import AutoencoderKL
            from transformers import T5EncoderModel, T5Tokenizer
            
            self.vae = AutoencoderKL.from_pretrained(
                f"{MODEL_CACHE}/Wan2.1-T2V-1.3B/vae",
                torch_dtype=torch.bfloat16
            ).to("cuda")
            
            self.text_encoder = T5EncoderModel.from_pretrained(
                f"{MODEL_CACHE}/Wan2.1-T2V-1.3B/text_encoder",
                torch_dtype=torch.bfloat16
            ).to("cuda")
            
            self.tokenizer = T5Tokenizer.from_pretrained(
                f"{MODEL_CACHE}/Wan2.1-T2V-1.3B/tokenizer"
            )
            
            # Whisper for audio processing
            import whisper
            self.whisper_model = whisper.load_model(
                "large-v3", 
                download_root=f"{MODEL_CACHE}/whisper-large-v3"
            )
            
            print("[✓] All models loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load models: {str(e)}")
            raise

    def predict(
        self,
        prompt: str = Input(
            description="Text description of the desired video",
            default="A person dancing to energetic music"
        ),
        image: CogPath = Input(
            description="Optional reference image for character/scene (for text+image or text+image+audio modes)",
            default=None
        ),
        audio: CogPath = Input(
            description="Optional audio file for synchronization (for text+audio or text+image+audio modes)", 
            default=None
        ),
        mode: str = Input(
            description="Generation mode based on available inputs",
            choices=["text_only", "text_image", "text_audio", "text_image_audio"],
            default="text_only"
        ),
        frames: int = Input(
            description="Number of frames to generate (HuMo is trained on 97-frame sequences)",
            default=97,
            ge=1,
            le=97
        ),
        height: int = Input(
            description="Video height in pixels",
            choices=[480, 720],
            default=720
        ),
        width: int = Input(
            description="Video width in pixels (recommended: 832 for 480p, 1280 for 720p)",
            choices=[832, 1280],
            default=1280
        ),
        steps: int = Input(
            description="Number of denoising steps (higher = better quality, slower generation)",
            default=50,
            ge=30,
            le=50
        ),
        scale_t: float = Input(
            description="Text guidance strength (higher = better text adherence)",
            default=1.0,
            ge=0.0,
            le=2.0
        ),
        scale_a: float = Input(
            description="Audio guidance strength (higher = better audio synchronization)",
            default=1.0,
            ge=0.0, 
            le=2.0
        ),
        seed: int = Input(
            description="Random seed for reproducible results. Use -1 for random seed",
            default=-1
        )
    ) -> CogPath:
        """Generate human-centric video using HuMo"""
        
        if seed == -1:
            seed = int(time.time())
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"[!] Generating video with mode: {mode}")
        print(f"[~] Resolution: {width}x{height}, Frames: {frames}, Steps: {steps}")
        print(f"[~] Text guidance: {scale_t}, Audio guidance: {scale_a}")
        print(f"[~] Seed: {seed}")
        
        try:
            # Validate inputs based on mode
            if mode in ["text_image", "text_image_audio"] and image is None:
                raise ValueError(f"Mode '{mode}' requires an input image")
            
            if mode in ["text_audio", "text_image_audio"] and audio is None:
                raise ValueError(f"Mode '{mode}' requires an input audio file")
            
            # Update config with user parameters
            self.config.generation.frames = frames
            self.config.generation.height = height
            self.config.generation.width = width  
            self.config.generation.scale_t = scale_t
            self.config.generation.scale_a = scale_a
            self.config.generation.mode = {
                "text_only": "T",
                "text_image": "TI", 
                "text_audio": "TA",
                "text_image_audio": "TIA"
            }[mode]
            self.config.diffusion.timesteps.sampling.steps = steps
            
            # Prepare inputs
            from humo.utils.inference_utils import prepare_inputs, generate_video
            
            inputs = prepare_inputs(
                prompt=prompt,
                image_path=str(image) if image else None,
                audio_path=str(audio) if audio else None,
                config=self.config,
                tokenizer=self.tokenizer,
                whisper_model=self.whisper_model if audio else None
            )
            
            print("[!] Starting video generation...")
            generation_start = time.time()
            
            # Generate video
            with torch.inference_mode():
                video_frames = generate_video(
                    model=self.model,
                    vae=self.vae, 
                    text_encoder=self.text_encoder,
                    inputs=inputs,
                    config=self.config,
                    device="cuda"
                )
            
            generation_time = time.time() - generation_start
            print(f"[✓] Video generation completed in {generation_time:.1f}s")
            
            # Save video
            output_path = Path(tempfile.mkdtemp()) / "output.mp4"
            
            from humo.utils.video_utils import save_video
            save_video(
                video_frames,
                str(output_path),
                fps=25,  # HuMo is trained at 25 FPS
                audio_path=str(audio) if audio and mode in ["text_audio", "text_image_audio"] else None
            )
            
            print(f"[✓] Video saved to {output_path}")
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            
            return CogPath(output_path)
            
        except Exception as e:
            print(f"[ERROR] Video generation failed: {str(e)}")
            torch.cuda.empty_cache()
            gc.collect()
            raise
