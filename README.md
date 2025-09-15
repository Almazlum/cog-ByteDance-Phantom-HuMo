# HuMo: Human-Centric Video Generation

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2509.08519-b31b1b.svg)](https://arxiv.org/abs/2509.08519)&nbsp;
[![Replicate](https://replicate.com/bytedance-research/humo/badge)](https://replicate.com/bytedance-research/humo)&nbsp;
[![Hugging Face](https://img.shields.io/static/v1?label=🤗%20Hugging%20Face&message=Model&color=orange)](https://huggingface.co/bytedance-research/HuMo)

> **HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning**  
> A unified framework for generating high-quality, controllable human videos from text, images, and audio.

<p align="center">
<img src="assets/teaser.png" width="95%">
</p>

## ✨ Key Features

HuMo supports three powerful generation modes:

- **Text + Image → Video**: Generate videos with custom character appearance, clothing, and scenes
- **Text + Audio → Video**: Create audio-synchronized videos from text descriptions and audio
- **Text + Image + Audio → Video**: Ultimate control combining all three modalities

## 🚀 Quick Start with Cog

This repository contains a [Cog](https://cog.run) implementation of HuMo, making it easy to run locally or deploy on [Replicate](https://replicate.com).

### Prerequisites

- **GPU**: NVIDIA H100 (recommended) or A100 with 40GB+ VRAM
- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
- **Cog**: [Install Cog](https://cog.run/docs/getting-started-own-model)

### Installation & Usage

```bash
# Clone the repository
git clone https://github.com/replicate/cog-ByteDance-Phantom-HuMo
cd cog-ByteDance-Phantom-HuMo

# Run a prediction (this will automatically build the container)
cog predict -i prompt="A person dancing to energetic music" \
             -i audio=@examples/audio_sample.wav \
             -i mode="text_audio"
```

That's it! The model will automatically:
1. Download the required weights (~104GB) from our CDN
2. Build the Docker container 
3. Generate your video

### Input Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `prompt` | string | Text description of the desired video | *Required* |
| `image` | file | Reference image for character/scene (optional) | None |
| `audio` | file | Audio file for synchronization (optional) | None |
| `mode` | string | Generation mode: `text_only`, `text_image`, `text_audio`, `text_image_audio` | `text_only` |
| `frames` | integer | Number of frames to generate (1-97) | 97 |
| `height` | integer | Video height in pixels (480 or 720) | 720 |
| `width` | integer | Video width in pixels (832 or 1280) | 1280 |
| `steps` | integer | Number of denoising steps (30-50) | 50 |
| `scale_t` | float | Text guidance strength (0.0-2.0) | 1.0 |
| `scale_a` | float | Audio guidance strength (0.0-2.0) | 1.0 |
| `seed` | integer | Random seed for reproducible results | None |

### Example Usage

**Text + Audio Generation:**
```bash
cog predict -i prompt="A professional dancer performing contemporary dance" \
             -i audio=@examples/dance_music.wav \
             -i mode="text_audio" \
             -i frames=97
```

**Text + Image + Audio Generation:**
```bash
cog predict -i prompt="A person singing passionately on stage" \
             -i image=@examples/reference_person.jpg \
             -i audio=@examples/singing_audio.wav \
             -i mode="text_image_audio"
```

## 📁 Repository Structure

```
├── predict.py          # Main Cog prediction interface
├── cog.yaml           # Cog configuration
├── requirements.txt   # Python dependencies
├── humo/             # Core HuMo model code
├── examples/         # Example inputs and test cases
└── assets/          # Documentation assets
```

## 🔧 Development

### Local Development Setup

```bash
# Clone and enter directory
git clone https://github.com/replicate/cog-ByteDance-Phantom-HuMo
cd cog-ByteDance-Phantom-HuMo

# Build the container
cog build

# Run predictions
cog predict -i prompt="Your text here" -i mode="text_only"
```

### Custom Configuration

The model behavior can be fine-tuned by modifying the generation parameters:

- **Higher `steps`** (40-50): Better quality, slower generation
- **Higher `scale_t`**: Stronger text adherence
- **Higher `scale_a`**: Better audio synchronization
- **720p resolution**: Best quality (requires more VRAM)
- **480p resolution**: Faster generation

## 🎯 Use Cases

**Content Creation:**
- Music videos and dance content
- Character-based storytelling
- Audio-visual presentations

**Entertainment:**
- Interactive avatar generation
- Video game character animation
- Film and media production

**Research & Education:**
- Human motion studies
- Audio-visual synchronization research
- Multimodal AI demonstrations

## ⚡ Performance

- **Generation Time**: ~2-5 minutes for 97 frames (4 seconds) on H100
- **Memory Requirements**: 40GB+ VRAM recommended
- **Resolution Support**: 480p, 720p
- **Frame Rate**: 25 FPS output

## 📝 Technical Details

HuMo is built on a 17B parameter transformer architecture with:
- **Multi-modal conditioning** for text, image, and audio inputs
- **Temporal consistency** across 97-frame sequences
- **High-fidelity synthesis** at up to 720p resolution
- **Efficient caching system** for fast model loading

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📜 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## 📚 Citation

If you use HuMo in your research, please cite:

```bibtex
@misc{chen2025humo,
      title={HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning}, 
      author={Liyang Chen and Tianxiang Ma and Jiawei Liu and Bingchuan Li and Zhuowei Chen and Lijie Liu and Xu He and Gen Li and Qian He and Zhiyong Wu},
      year={2025},
      eprint={2509.08519},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 🔗 Links

- **Paper**: [arXiv:2509.08519](https://arxiv.org/abs/2509.08519)
- **Project Page**: [phantom-video.github.io/HuMo](https://phantom-video.github.io/HuMo/)
- **Original Repository**: [ByteDance Research](https://huggingface.co/bytedance-research/HuMo)
- **Replicate Demo**: [replicate.com/bytedance-research/humo](https://replicate.com/bytedance-research/humo)

---

For questions or support, please open an issue or visit our [project page](https://phantom-video.github.io/HuMo/).
