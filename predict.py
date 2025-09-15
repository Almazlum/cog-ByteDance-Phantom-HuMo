"""
HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning
Cog implementation for Replicate deployment
"""

import os
import sys
import gc
import subprocess
import time
import json
import tempfile
from pathlib import Path
from typing import Optional

# GPU optimizations
os.environ.update({
    "CUDA_LAUNCH_BLOCKING": "0",
    "TORCH_BACKENDS_CUDNN_BENCHMARK": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
    "OMP_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
    "TOKENIZERS_PARALLELISM": "false"
})

# Model cache configuration
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
            print(f"[✓] {extracted_name} already exists")

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
        
        print("[✓] HuMo setup complete - using generate.py for inference")
        self.model_loaded = True

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
            
            # Create temporary directory for processing
            temp_dir = Path(tempfile.mkdtemp())
            output_path = temp_dir / "output.mp4"
            
            # Create test case configuration
            test_case = {
                "case_1": {
                    "prompt": prompt,
                    "img_paths": [str(image)] if image else [],
                    "audio_path": str(audio) if audio else ""
                }
            }
            
            test_case_path = temp_dir / "test_case.json"
            with open(test_case_path, "w") as f:
                json.dump(test_case, f, indent=2)
            
            # Set generation parameters via environment or config modification
            os.environ.update({
                "HUMO_FRAMES": str(frames),
                "HUMO_HEIGHT": str(height),
                "HUMO_WIDTH": str(width),
                "HUMO_STEPS": str(steps),
                "HUMO_SCALE_T": str(scale_t),
                "HUMO_SCALE_A": str(scale_a),
                "HUMO_SEED": str(seed)
            })
            
            print("[!] Starting video generation using HuMo generate.py...")
            generation_start = time.time()
            
            # Determine which generate script to use based on mode
            if mode == "text_audio" or (mode == "text_image_audio" and audio):
                generate_script = "humo/generate.py"
            else:
                generate_script = "humo/generate.py"  # Default to main script
            
            # Run HuMo generation
            cmd = [
                "python", generate_script,
                "--config_path", "humo/configs/inference/generate.yaml",
                "--test_case_path", str(test_case_path),
                "--output_dir", str(temp_dir),
                "--case_name", "case_1"
            ]
            
            print(f"[~] Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
            
            if result.returncode != 0:
                print(f"[ERROR] Generation failed:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                raise RuntimeError(f"HuMo generation failed with return code {result.returncode}")
            
            generation_time = time.time() - generation_start
            print(f"[✓] Video generation completed in {generation_time:.1f}s")
            
            # Find the generated video file
            generated_files = list(temp_dir.glob("*.mp4"))
            if not generated_files:
                raise RuntimeError("No video file was generated")
            
            generated_video = generated_files[0]
            final_output = temp_dir / "humo_output.mp4"
            generated_video.rename(final_output)
            
            print(f"[✓] Video saved to {final_output}")
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            
            return CogPath(final_output)
            
        except Exception as e:
            print(f"[ERROR] Video generation failed: {str(e)}")
            torch.cuda.empty_cache()
            gc.collect()
            raise
