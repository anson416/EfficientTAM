# Architectural Patterns

This document describes recurring design patterns used across the EfficientTAM codebase.

## 1. Hydra Configuration-Driven Module Composition

**Where**: `build_efficienttam.py`, `efficienttam_base.py`, all `configs/efficienttam/*.yaml` files

**Pattern**: Models are composed from YAML configuration files using Hydra's instantiation system. Each component (encoder, decoder, memory modules) is specified declaratively.

**Example Structure**:
```yaml
image_encoder:
  _target_: modeling.image_encoder.ImageEncoder
  trunk:
    _target_: modeling.backbones.vitdet.ViT
  neck:
    _target_: modeling.backbones.vitdet.ViTDetNeck
```

**Implementation**: `build_efficienttam.py:50-150` - `build_efficienttam()` function uses `hydra_cfg.instantiate()`

**Benefit**: Create model variants without code changes; configuration is single source of truth

## 2. Stateful Interactive Inference

**Where**: `efficienttam_video_predictor.py`, `efficienttam_image_predictor.py`

**Pattern**: Predictors maintain mutable inference state dictionaries that accumulate user interactions and computed features across method calls.

**Key Methods**:
- `init_state()` - Initialize video/frames in state: `efficienttam_video_predictor.py:150-250`
- `add_new_points_or_box()` - Add prompts: `efficienttam_video_predictor.py:400-600`
- `propagate_in_video()` - Run inference: `efficienttam_video_predictor.py:800-1000`

**State Dictionary Contains**:
- Video frames and metadata
- Image encoder features (cached per frame)
- Memory features from previous frames
- Object tracking information (masks, points, boxes)
- Conditioning frames for each object

**Example**: `efficienttam_video_predictor.py:700-750` - State dict structure

**Benefit**: Enables incremental, interactive workflows where users iteratively refine results

## 3. Multi-Scale Feature Pyramid (FPN)

**Where**: `modeling/backbones/image_encoder.py`, `modeling/backbones/vitdet.py`

**Pattern**: Vision Transformer outputs are projected to multiple resolution levels using 1x1 convolutions, creating a Feature Pyramid Network.

**Implementation**:
- `vitdet.py:300-400` - `ViTDetNeck` class
- `image_encoder.py:100-200` - FPN construction

**Output**: `backbone_fpn` dict with keys like `vision_features`, `vision_pos_enc`, containing features at different scales

**Benefit**: Multi-scale understanding for objects of varying sizes

## 4. Temporal Positional Encoding

**Where**: `efficienttam_base.py:400-600`, `memory_encoder.py`, `memory_attention.py`

**Pattern**: Frame-to-frame temporal relationships encoded using sinusoidal positional embeddings based on temporal distance.

**Components**:
- `maskmem_tpos_enc` - Sinusoidal encoding of temporal offsets
- `no_mem_embed` - Special learnable embedding for initial frame (no memory)
- Temporal distance calculation: current_frame_idx - memory_frame_idx

**Example**: `efficienttam_base.py:500-550` - Temporal encoding in `_prepare_memory_conditioned_features()`

**Benefit**: Model learns temporal dependencies without explicit recurrence

## 5. Object ID Bidirectional Mapping

**Where**: `efficienttam_video_predictor.py:100-200`, used throughout inference methods

**Pattern**: Maintain bidirectional mappings between user-facing object IDs (integers) and internal tensor indices.

**Data Structures**:
- `obj_id_to_idx`: Dict mapping user IDs → internal indices
- `obj_idx_to_id`: Dict mapping internal indices → user IDs
- `obj_ids`: OrderedDict preserving insertion order

**Example**: `efficienttam_video_predictor.py:150-200` - Mapping initialization and updates

**Benefit**: Decouples user interface from internal tensor operations; allows arbitrary user-chosen IDs

## 6. Memory Offloading for Edge Deployment

**Where**: `efficienttam_video_predictor.py:50-100`, throughout inference methods

**Pattern**: Explicit device management with CPU offloading options to reduce GPU memory footprint.

**Options** (constructor parameters):
- `offload_video_to_cpu` - Store video frames on CPU, transfer on-demand
- `offload_state_to_cpu` - Store inference state features on CPU
- `async_loading_frames` - **Line 53** - Parallel frame loading with thread pooling

**Implementation**: `efficienttam_video_predictor.py:200-300` - Device transfer logic in `_get_image_feature()`

**Recent Optimization**: `PrefetchVideoFrameLoader` with tunable prefetch/cache sizes for memory-speed tradeoffs

**Benefit**: Enables deployment on devices with limited GPU memory (mobile, edge devices)

## 7. SAM-Style Prompt Encoding

**Where**: `modeling/sam/prompt_encoder.py`, `modeling/sam/mask_decoder.py`

**Pattern**: Separate encoding of different prompt types (points, boxes, masks) into embedding space, followed by transformer-based decoding.

**Prompt Types**:
- Points: Positional embedding + learned positive/negative embeddings
- Boxes: Encoded as 4 corner points
- Masks: Downsampled and embedded with convolutions
- "No mask" indicator: Learnable embedding when no mask provided

**Integration**: `efficienttam_base.py:700-900` - `_forward_sam_heads()` combines prompt embeddings with image features

**Benefit**: Flexible multi-modal prompting; easy to add new prompt types

## 8. Two-Way Transformer Decoder

**Where**: `modeling/sam/transformer.py`

**Pattern**: Alternating self-attention and cross-attention layers with residual connections.

**Structure per layer**:
1. Self-attention on queries
2. Cross-attention: queries attend to keys
3. Feed-forward network
4. Layer normalization after each

**Variants**:
- `TwoWayTransformer` - Standard implementation
- Multiple `RoPEAttention` variants (standard, efficient1, efficient2) with Rotary Position Embeddings

**Example**: `transformer.py:200-400` - `TwoWayAttentionBlock` class

**Benefit**: Deep feature fusion between prompts and image features

## 9. Mask Post-Processing Pipeline

**Where**: `utils/misc.py:200-400`, `efficienttam_image_predictor.py:300-400`

**Pattern**: Chain of transformations to clean up predicted masks.

**Steps**:
1. Sigmoid activation → probabilities
2. Threshold → binary masks
3. Connected components (CUDA kernel in `csrc/connected_components.cu`)
4. Hole filling - Remove small holes inside masks
5. Sprinkle removal - Remove small disconnected regions

**Parameters**:
- `fill_hole_area` - Max hole size to fill
- `remove_small_region_area` - Min region size to keep

**Implementation**: `utils/misc.py:250-350` - `_process_mask_batch()`

**Benefit**: Cleaner output masks, removes noise artifacts

## 10. Memory Cross-Attention Mechanism

**Where**: `memory_attention.py`, used in `efficienttam_base.py:600-800`

**Pattern**: Efficient cross-attention where current frame features attend to memory bank of previous frames.

**Components**:
- Memory bank: Stack of encoded features from previous frames + temporal encodings
- Query: Current frame features
- Keys/Values: Memory bank features
- Output: Temporally-enriched current frame features

**Efficiency Optimization**: `memory_attention.py:50-150` - Specialized attention implementations vs. standard attention

**Example**: `efficienttam_base.py:650-700` - `_prepare_memory_conditioned_features()` applies memory attention

**Benefit**: Captures temporal context without full recurrence; better than frame-by-frame processing

## 11. Automatic Mask Generation (Grid Sampling)

**Where**: `automatic_mask_generator.py`

**Pattern**: Dense grid-based point sampling followed by mask prediction and post-filtering.

**Algorithm**:
1. Generate grid of points across image: `automatic_mask_generator.py:200-250`
2. Predict masks for each point
3. Calculate stability scores (IoU at different thresholds)
4. Filter low-quality masks
5. Non-maximum suppression to remove duplicates
6. Sort by predicted IoU score

**Parameters**: `points_per_side`, `pred_iou_thresh`, `stability_score_thresh`

**Benefit**: Zero-shot "segment everything" without user interaction

## 12. Cached Feature Computation

**Where**: `efficienttam_video_predictor.py:250-350`, `efficienttam_image_predictor.py:150-200`

**Pattern**: Cache expensive computations (image encoder features) keyed by frame index.

**Implementation**:
- Check if features exist in state: `if frame_idx not in state["cached_features"]`
- Compute if missing: `features = self.model.forward_image(frame)`
- Store in state dict: `state["cached_features"][frame_idx] = features`

**Example**: `efficienttam_video_predictor.py:280-320` - `_get_image_feature()`

**Benefit**: Avoid redundant encoder passes when user adds prompts to same frame

## 13. Gradient Checkpointing

**Where**: `modeling/backbones/vitdet.py`, `modeling/sam/transformer.py`

**Pattern**: Use PyTorch's activation checkpointing to trade compute for memory during training.

**Implementation**: Controlled via Hydra config, applied to ViT blocks and transformer layers

**Benefit**: Train larger models or with larger batch sizes on same GPU memory

## Common Conventions

- **Device handling**: Explicit `.to(device)` calls, avoid implicit device transfers
- **Batching**: Most operations support batch dimension even for single-sample inference
- **Type hints**: Extensive use of Python type annotations
- **Configuration**: All hyperparameters exposed via Hydra YAML configs
- **Assertions**: Liberal use of `assert` for shape and type validation in development
