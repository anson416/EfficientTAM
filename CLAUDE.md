# EfficientTAM

## What

EfficientTAM is an efficient foundation model for promptable image and video segmentation that achieves comparable performance to SAM 2 while running at >10 FPS on iPhone 15. Users provide prompts (clicks, boxes, or masks) to segment and track objects across video frames or segment objects in images.

## Tech Stack

- **Core**: PyTorch 2.5.1+, torchvision, CUDA extensions
- **Configuration**: Hydra 1.3.2 (YAML-based model configs)
- **Video Processing**: decord, moviepy, OpenCV
- **UI**: Gradio 4.44.0 (web demos), Flask (backend)
- **Build**: setuptools with CUDA C++ extension compilation

## Project Structure

```
efficient_track_anything/       # Main package
├── modeling/                   # Neural network models
│   ├── backbones/             # ViT-based image encoders
│   ├── sam/                   # SAM components (mask decoder, prompt encoder, transformer)
│   ├── efficienttam_base.py   # Base model class (1023 lines)
│   ├── memory_attention.py    # Temporal cross-attention modules
│   ├── memory_encoder.py      # Encodes masks into memory tokens
│   └── position_encoding.py   # RoPE positional encodings
├── utils/                     # Utilities (video loading, mask operations, transforms)
├── configs/efficienttam/      # 8 model variant configs (ti/s, various resolutions)
├── csrc/                      # CUDA extensions (connected_components.cu)
├── efficienttam_video_predictor.py    # Video inference engine (1352 lines)
├── efficienttam_image_predictor.py    # Image segmentation (472 lines)
├── build_efficienttam.py              # Model factory/builder
├── automatic_mask_generator.py        # "Segment everything" mode
└── benchmark.py                       # Performance benchmarking

notebooks/                      # Example scripts
app.py / app_image.py          # Gradio web demos
checkpoints/                   # Model weights directory
```

**Key Files:**
- `efficienttam_video_predictor.py:1-1352` - Main video inference with interactive state management
- `efficienttam_base.py:1-1023` - Base model combining encoder, memory, decoder
- `modeling/sam/transformer.py` - Two-way transformer and RoPE attention variants
- `build_efficienttam.py:1-182` - Hydra-based model instantiation

## Essential Commands

**Installation:**
```bash
conda create -n efficient_track_anything python=3.12
conda activate efficient_track_anything
pip install -e .  # Builds CUDA extensions
```

**Download Checkpoints:**
```bash
cd checkpoints && ./download_checkpoints.sh && cd ..
```

**Run Demos:**
```bash
python app.py              # Video tracking demo
python app_image.py        # Image segmentation demo
```

**Run Examples:**
```bash
python notebooks/example_video.py              # Video tracking
python notebooks/example_image.py              # Image segmentation
python notebooks/example_segment_everything.py # Automatic masks
```

**Benchmark Performance:**
```bash
python efficient_track_anything/benchmark.py
```

**Programmatic Usage:**
```python
from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor

predictor = build_efficienttam_video_predictor(
    "configs/efficienttam/efficienttam_s.yaml",
    "./checkpoints/efficienttam_s.pt"
)
```

**Build Options:**
```bash
Efficient_Track_Anything_BUILD_CUDA=0 pip install -e .  # Skip CUDA extensions
```

## Model Variants

8 configurations available in `configs/efficienttam/`:
- **EfficientTAM-S**: ViT-384 backbone, 1024x1024 input (`efficienttam_s.yaml`)
- **EfficientTAM-Ti**: ViT-192 backbone, 1024x1024 input (`efficienttam_ti.yaml`)
- Variants: `_512x512` (faster), `_1`, `_2` (different attention mechanisms)

## Core Architecture

**Video Inference Pipeline:**
1. User input (clicks/boxes) → `add_new_points_or_box()` in `efficienttam_video_predictor.py:400-600`
2. Image encoder (ViT) → multi-scale features
3. Memory attention → temporal context fusion (`memory_attention.py:1-184`)
4. SAM prompt encoder → encode prompts (`modeling/sam/prompt_encoder.py`)
5. Mask decoder (two-way transformer) → output masks (`modeling/sam/mask_decoder.py`)
6. Post-processing → threshold, fill holes (`utils/misc.py:200-400`)
7. Memory encoder → store for next frame (`memory_encoder.py:1-185`)

**Key Capabilities:**
- Point-based and box-based prompting
- Multi-object tracking across video frames
- Interactive refinement (add clicks to improve masks)
- Automatic "segment everything" mode
- CPU/GPU offloading for edge devices (`efficienttam_video_predictor.py:53` - `async_loading_frames`)

## Memory Optimization (NEW)

EfficientTAM now supports fine-grained memory configuration for processing long videos with multiple objects:

**Key Parameters:**
- `offload_video_to_cpu=True` - Move frame cache to CPU RAM (saves ~384 MB GPU, -3% FPS)
- `prefetch_count=8` - Frames to prefetch ahead (affects CPU RAM: ~12 MB/frame)
- `cache_size=16` - Max frames in cache (affects CPU RAM: ~12 MB/frame)
- `max_kept_frames=16` - Output window size (affects GPU: ~1.6 MB/object/frame)

**Quick Start:**
```python
# Balanced configuration (recommended)
predictor = build_efficienttam_video_predictor(
    "configs/efficienttam/efficienttam_s.yaml",
    "./checkpoints/efficienttam_s.pt",
    max_kept_frames=16,
)
state = predictor.init_state(
    video_path,
    offload_video_to_cpu=True,
    async_loading_frames=True,
    prefetch_count=8,
    cache_size=16,
)
```

**See:** `MEMORY_OPTIMIZATION.md` for detailed guide and `notebooks/example_memory_optimization.py` for usage examples.

## Additional Documentation

When working on specific aspects, consult:
- `.claude/docs/architectural_patterns.md` - Design patterns, state management, attention mechanisms, memory optimization
- `MEMORY_OPTIMIZATION.md` - Complete guide to memory configuration parameters

## Recent Development

Recent work added configurable memory parameters (`prefetch_count`, `cache_size`, `max_kept_frames`) to enable efficient processing of long videos with multiple objects on limited VRAM.
