# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import atexit
import os
import threading
import time
import warnings
import weakref
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Thread
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def get_sdpa_settings():
    if torch.cuda.is_available():
        old_gpu = torch.cuda.get_device_properties(0).major < 7
        # only use Flash Attention on Ampere (8.0) or newer GPUs
        use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
        if not use_flash_attn:
            warnings.warn(
                "Flash Attention is disabled as it requires a GPU with Ampere (8.0) CUDA capability.",
                category=UserWarning,
                stacklevel=2,
            )
        # keep math kernel for PyTorch versions before 2.2 (Flash Attention v2 is only
        # available on PyTorch 2.2+, while Flash Attention v1 cannot handle all cases)
        pytorch_version = tuple(
            int(v) for v in torch.__version__.split(".")[:2]
        )
        if pytorch_version < (2, 2):
            warnings.warn(
                f"You are using PyTorch {torch.__version__} without Flash Attention v2 support. "
                "Consider upgrading to PyTorch 2.2+ for Flash Attention v2 (which could be faster).",
                category=UserWarning,
                stacklevel=2,
            )
        math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
    else:
        old_gpu = True
        use_flash_attn = False
        math_kernel_on = True

    return old_gpu, use_flash_attn, math_kernel_on


def get_connected_components(mask):
    """
    Get the connected components (8-connectivity) of binary masks of shape (N, 1, H, W).

    Inputs:
    - mask: A binary mask tensor of shape (N, 1, H, W), where 1 is foreground and 0 is
            background.

    Outputs:
    - labels: A tensor of shape (N, 1, H, W) containing the connected component labels
              for foreground pixels and 0 for background pixels.
    - counts: A tensor of shape (N, 1, H, W) containing the area of the connected
              components for foreground pixels and 0 for background pixels.
    """
    from efficient_track_anything import _C

    return _C.get_connected_componnets(mask.to(torch.uint8).contiguous())


def mask_to_box(masks: torch.Tensor):
    """
    compute bounding box given an input mask

    Inputs:
    - masks: [B, 1, H, W] masks, dtype=torch.Tensor

    Returns:
    - box_coords: [B, 1, 4], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.Tensor
    """
    B, _, h, w = masks.shape
    device = masks.device
    xs = torch.arange(w, device=device, dtype=torch.int32)
    ys = torch.arange(h, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None, ...].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None, ...].expand(B, 1, h, w)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, w).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, h).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    bbox_coords = torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)

    return bbox_coords


def _load_img_as_tensor(img_path, image_size):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(
            f"Unknown image dtype: {img_np.dtype} on {img_path}"
        )
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size  # the original video size
    return img, video_height, video_width


@dataclass
class FrameMetadata:
    """Immutable frame loading configuration."""

    img_paths: Tuple[str, ...]
    image_size: int
    img_mean: torch.Tensor
    img_std: torch.Tensor


def _load_frame_standalone(
    metadata: FrameMetadata, index: int
) -> Tuple[torch.Tensor, int, int]:
    """
    Standalone function for loading frames.
    Doesn't reference the loader instance - avoids GC/lifecycle issues.
    """
    img_pil = Image.open(metadata.img_paths[index])
    img_np = np.array(
        img_pil.convert("RGB").resize(
            (metadata.image_size, metadata.image_size)
        )
    )

    if img_np.dtype == np.uint8:
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype}")

    img = torch.from_numpy(img_np).permute(2, 0, 1).float()
    video_width, video_height = img_pil.size

    # Normalize
    img = img - metadata.img_mean
    img = img / metadata.img_std

    return img, video_height, video_width


class PrefetchVideoFrameLoader:
    """
    Thread-safe video frame loader with prefetching.

    Thread safety guarantees:
    - Single lock for all shared state (simpler, no deadlock possible)
    - Atomic check-and-load pattern
    - Safe cleanup with proper executor shutdown
    """

    # Class-level registry for cleanup on interpreter shutdown
    _instances: weakref.WeakSet = weakref.WeakSet()

    def __init__(
        self,
        img_paths,
        image_size: int,
        offload_video_to_cpu: bool,
        img_mean: torch.Tensor,
        img_std: torch.Tensor,
        compute_device: torch.device,
        prefetch_count: int = 16,
        cache_size: int = 32,
        num_workers: int = 4,
    ):
        # Immutable configuration (safe to access from any thread)
        self.metadata = FrameMetadata(
            img_paths=tuple(img_paths),
            image_size=image_size,
            img_mean=img_mean.clone(),
            img_std=img_std.clone(),
        )
        self.offload_video_to_cpu = offload_video_to_cpu
        self.compute_device = compute_device
        self.prefetch_count = prefetch_count
        self.cache_size = cache_size

        # Single lock for all mutable state - eliminates deadlock risk
        self._lock = threading.RLock()  # RLock allows reentrant acquisition

        # Protected by self._lock
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._prefetch_futures: Dict[int, Future] = {}
        self._loading_in_progress: set = set()  # Frames currently being loaded
        self._last_accessed_idx: int = -1
        self._prefetch_direction: int = 1
        self._shutdown: bool = False

        # Video dimensions (set once, then read-only)
        self.video_height: Optional[int] = None
        self.video_width: Optional[int] = None

        # Thread pool
        self._executor = ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="frame_loader"
        )

        # Register for cleanup
        PrefetchVideoFrameLoader._instances.add(self)

        # Load first frame synchronously to get dimensions
        first_frame = self._get_frame_blocking(0)
        with self._lock:
            self._cache[0] = first_frame

    def _get_frame_blocking(self, index: int) -> torch.Tensor:
        """Load a frame synchronously. Does not use cache."""
        img, h, w = _load_frame_standalone(self.metadata, index)

        if self.video_height is None:
            self.video_height = h
            self.video_width = w

        return img

    def _try_get_from_cache(self, index: int) -> Optional[torch.Tensor]:
        """Try to get frame from cache. Returns None if not cached."""
        with self._lock:
            if index in self._cache:
                self._cache.move_to_end(index)
                return self._cache[index]
        return None

    def _wait_for_prefetch(self, index: int) -> Optional[torch.Tensor]:
        """Wait for a prefetch future if one exists. Returns None if no future."""
        future = None
        with self._lock:
            if index in self._prefetch_futures:
                future = self._prefetch_futures.pop(index)

        if future is not None:
            try:
                img, h, w = future.result(
                    timeout=30.0
                )  # Timeout prevents infinite hang
                if self.video_height is None:
                    self.video_height = h
                    self.video_width = w
                return img
            except Exception as e:
                # Future failed, will fall back to sync load
                print(f"Prefetch failed for frame {index}: {e}")
                return None
        return None

    def _add_to_cache(self, index: int, img: torch.Tensor) -> None:
        """Add frame to cache with LRU eviction."""
        with self._lock:
            self._cache[index] = img
            self._cache.move_to_end(index)
            self._loading_in_progress.discard(index)

            # Evict oldest frames
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def _should_load(self, index: int) -> bool:
        """Check if we should load this frame (not cached, not loading)."""
        with self._lock:
            return (
                index not in self._cache
                and index not in self._prefetch_futures
                and index not in self._loading_in_progress
            )

    def _mark_loading(self, index: int) -> bool:
        """
        Mark frame as loading. Returns True if successfully marked.
        Returns False if already loading/cached.
        """
        with self._lock:
            if self._shutdown:
                return False
            if index in self._cache or index in self._loading_in_progress:
                return False
            self._loading_in_progress.add(index)
            return True

    def _submit_prefetch(self, index: int) -> None:
        """Submit a prefetch task for the given index."""
        if not (0 <= index < len(self.metadata.img_paths)):
            return

        with self._lock:
            if self._shutdown:
                return
            if index in self._cache or index in self._prefetch_futures:
                return
            if index in self._loading_in_progress:
                return

            try:
                future = self._executor.submit(
                    _load_frame_standalone, self.metadata, index
                )
                self._prefetch_futures[index] = future
            except RuntimeError:
                # Executor is shut down
                pass

    def _start_prefetch_batch(self, current_idx: int, direction: int) -> None:
        """Start prefetching frames in the given direction."""
        for offset in range(1, self.prefetch_count + 1):
            idx = current_idx + offset * direction
            self._submit_prefetch(idx)

        # Also prefetch a few in the opposite direction for direction changes
        for offset in range(1, min(4, self.prefetch_count)):
            idx = current_idx - offset * direction
            self._submit_prefetch(idx)

    def _update_access_pattern(self, index: int) -> int:
        """Update and return the predicted prefetch direction."""
        with self._lock:
            if self._last_accessed_idx >= 0:
                if index > self._last_accessed_idx:
                    self._prefetch_direction = 1
                elif index < self._last_accessed_idx:
                    self._prefetch_direction = -1
            self._last_accessed_idx = index
            return self._prefetch_direction

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get a frame by index. Thread-safe."""
        if not (0 <= index < len(self.metadata.img_paths)):
            raise IndexError(f"Frame index {index} out of range")

        # Update access pattern and get prefetch direction
        direction = self._update_access_pattern(index)

        # Try cache first
        img = self._try_get_from_cache(index)
        if img is not None:
            self._start_prefetch_batch(index, direction)
            return self._to_device(img)

        # Try waiting for prefetch
        img = self._wait_for_prefetch(index)
        if img is not None:
            self._add_to_cache(index, img)
            self._start_prefetch_batch(index, direction)
            return self._to_device(img)

        # Load synchronously (with loading guard to prevent duplicate work)
        if self._mark_loading(index):
            try:
                img = self._get_frame_blocking(index)
                self._add_to_cache(index, img)
            except Exception:
                with self._lock:
                    self._loading_in_progress.discard(index)
                raise
        else:
            # Another thread is loading, wait for it
            while True:
                img = self._try_get_from_cache(index)
                if img is not None:
                    break
                # Small sleep to avoid busy waiting
                time.sleep(0.001)

        self._start_prefetch_batch(index, direction)
        return self._to_device(img)

    def _to_device(self, img: torch.Tensor) -> torch.Tensor:
        """Transfer to compute device if needed."""
        if not self.offload_video_to_cpu:
            return img.to(self.compute_device, non_blocking=True)
        return img

    def __len__(self) -> int:
        return len(self.metadata.img_paths)

    def shutdown(
        self, wait: bool = True, timeout: Optional[float] = 5.0
    ) -> None:
        """
        Explicitly shut down the loader.

        Args:
            wait: If True, wait for pending tasks to complete
            timeout: Maximum time to wait (only used if wait=True)
        """
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True

            # Cancel pending futures
            futures_to_cancel = list(self._prefetch_futures.values())
            self._prefetch_futures.clear()

        # Cancel futures outside lock
        for future in futures_to_cancel:
            future.cancel()

        # Shutdown executor
        self._executor.shutdown(wait=wait, cancel_futures=True)

        # Clear cache
        with self._lock:
            self._cache.clear()
            self._loading_in_progress.clear()

    def __del__(self):
        """Destructor - attempt graceful cleanup."""
        try:
            self.shutdown(wait=False)
        except Exception:
            pass  # Ignore errors in destructor

    @classmethod
    def shutdown_all(cls, wait: bool = False) -> None:
        """Shutdown all loader instances. Called on interpreter exit."""
        for instance in list(cls._instances):
            try:
                instance.shutdown(wait=wait)
            except Exception:
                pass


# Register cleanup on interpreter shutdown
atexit.register(PrefetchVideoFrameLoader.shutdown_all)


class AsyncVideoFrameLoader:
    """
    A list of video frames to be load asynchronously without blocking session start.
    """

    def __init__(
        self,
        img_paths,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        compute_device,
    ):
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        # items in `self.images` will be loaded asynchronously
        self.images = [None] * len(img_paths)
        # catch and raise any exceptions in the async loading thread
        self.exception = None
        # video_height and video_width be filled when loading the first image
        self.video_height = None
        self.video_width = None
        self.compute_device = compute_device

        # load the first frame to fill video_height and video_width and also
        # to cache it (since it's most likely where the user will click)
        self.__getitem__(0)

        # # load the rest of frames asynchronously without blocking the session start
        # def _load_frames():
        #     try:
        #         for n in tqdm(
        #             range(len(self.images)), desc="frame loading (JPEG)"
        #         ):
        #             self.__getitem__(n)
        #     except Exception as e:
        #         self.exception = e

        # self.thread = Thread(target=_load_frames, daemon=True)
        # self.thread.start()

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError(
                "Failure in frame loading thread"
            ) from self.exception

        img = self.images[index]
        if img is not None:
            return img

        img, video_height, video_width = _load_img_as_tensor(
            self.img_paths[index], self.image_size
        )
        self.video_height = video_height
        self.video_width = video_width
        # normalize by mean and std
        img -= self.img_mean
        img /= self.img_std
        if not self.offload_video_to_cpu:
            img = img.to(self.compute_device, non_blocking=True)
        # self.images[index] = img
        return img

    def __len__(self):
        return len(self.images)


def load_video_frames(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda"),
):
    """
    Load the video frames from video_path. The frames are resized to image_size as in
    the model and are loaded to GPU if offload_video_to_cpu=False. This is used by the demo.
    """
    is_bytes = isinstance(video_path, bytes)
    is_str = isinstance(video_path, str)
    is_mp4_path = is_str and os.path.splitext(video_path)[-1] in [
        ".mp4",
        ".MP4",
    ]
    if is_bytes or is_mp4_path:
        return load_video_frames_from_video_file(
            video_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            compute_device=compute_device,
        )
    elif is_str and os.path.isdir(video_path):
        return load_video_frames_from_jpg_images(
            video_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
        )
    else:
        raise NotImplementedError(
            "Only MP4 video and JPEG folder are supported at this moment"
        )


def load_video_frames_from_jpg_images(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda"),
    # New parameters
    prefetch_count=16,
    cache_size=32,
):
    """
    Load the video frames from a directory of JPEG files ("<frame_index>.jpg" format).

    The frames are resized to image_size x image_size and are loaded to GPU if
    `offload_video_to_cpu` is `False` and to CPU if `offload_video_to_cpu` is `True`.

    You can load a frame asynchronously by setting `async_loading_frames` to `True`.
    """
    if isinstance(video_path, str) and os.path.isdir(video_path):
        jpg_folder = video_path
    else:
        raise NotImplementedError(
            "Only JPEG frames are supported at this moment. For video files, you may use "
            "ffmpeg (https://ffmpeg.org/) to extract frames into a folder of JPEG files, such as \n"
            "```\n"
            "ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'\n"
            "```\n"
            "where `-q:v` generates high-quality JPEG frames and `-start_number 0` asks "
            "ffmpeg to start the JPEG file from 00000.jpg."
        )

    frame_names = [
        p
        for p in os.listdir(jpg_folder)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    num_frames = len(frame_names)
    if num_frames == 0:
        raise RuntimeError(f"no images found in {jpg_folder}")
    img_paths = [
        os.path.join(jpg_folder, frame_name) for frame_name in frame_names
    ]
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    if async_loading_frames:
        loader = PrefetchVideoFrameLoader(
            img_paths=img_paths,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            compute_device=compute_device,
            prefetch_count=prefetch_count,
            cache_size=cache_size,
        )
        return loader, loader.video_height, loader.video_width

    def _load_one(idx: int, path: str):
        img, h, w = _load_img_as_tensor(path, image_size)
        return idx, img, h, w

    images = torch.zeros(
        num_frames, 3, image_size, image_size, dtype=torch.float32
    )
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)

    video_height, video_width = None, None
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_load_one, i, p) for i, p in enumerate(img_paths)]
        try:
            for fut in tqdm(
                as_completed(futures), total=num_frames, desc="Loading frames"
            ):
                idx, img, h, w = fut.result()
                if video_height is None:
                    video_height, video_width = h, w
                if not offload_video_to_cpu:
                    img = img.to(compute_device)
                images[idx] = img
        except Exception:
            for f in futures:
                f.cancel()
            raise

    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


def load_video_frames_from_video_file(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    compute_device=torch.device("cuda"),
):
    """Load the video frames from a video file."""
    import decord

    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    # Get the original video height and width
    decord.bridge.set_bridge("torch")
    video_height, video_width, _ = decord.VideoReader(video_path).next().shape
    # Iterate over all frames in the video
    images = []
    for frame in decord.VideoReader(
        video_path, width=image_size, height=image_size
    ):
        images.append(frame.permute(2, 0, 1))

    images = torch.stack(images, dim=0).float() / 255.0
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


def fill_holes_in_mask_scores(mask, max_area):
    """
    A post processor to fill small holes in mask scores with area under `max_area`.
    """
    # Holes are those connected components in background with area <= self.max_area
    # (background regions are those with mask scores <= 0)
    assert max_area > 0, "max_area must be positive"

    input_mask = mask
    try:
        labels, areas = get_connected_components(mask <= 0)
        is_hole = (labels > 0) & (areas <= max_area)
        # We fill holes with a small positive mask score (0.1) to change them to foreground.
        mask = torch.where(is_hole, 0.1, mask)
    except Exception as e:
        # Following SAM 2, skip the post-processing step on removing small holes if the CUDA kernel fails
        warnings.warn(
            f"{e}\n\nSkipping the post-processing step due to the error above. You can "
            "still use Efficient Track Anything and it's OK to ignore the error above.",
            category=UserWarning,
            stacklevel=2,
        )
        mask = input_mask

    return mask


def concat_points(old_point_inputs, new_points, new_labels):
    """Add new points and labels to previous point inputs (add at the end)."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat(
            [old_point_inputs["point_coords"], new_points], dim=1
        )
        labels = torch.cat(
            [old_point_inputs["point_labels"], new_labels], dim=1
        )

    return {"point_coords": points, "point_labels": labels}
