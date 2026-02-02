#!/usr/bin/env python3
"""Test script to verify the new memory configuration parameters work correctly."""

import torch
from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor


def test_build_with_params():
    """Test that the new parameters can be passed to the builder."""
    print("Testing build_efficienttam_video_predictor with custom parameters...")

    # Build with custom max_kept_frames
    try:
        predictor = build_efficienttam_video_predictor(
            "configs/efficienttam/efficienttam_ti.yaml",
            ckpt_path=None,  # Don't load checkpoint for this test
            device="cpu",     # Use CPU to avoid CUDA requirements
            max_kept_frames=8,  # Custom value instead of default 16
        )
        print(f"✓ Successfully built predictor with max_kept_frames={predictor.max_kept_frames}")
        assert predictor.max_kept_frames == 8, f"Expected max_kept_frames=8, got {predictor.max_kept_frames}"
        print("✓ max_kept_frames parameter correctly set")
    except Exception as e:
        print(f"✗ Failed to build predictor: {e}")
        raise

    print("\nAll tests passed!")


def test_init_state_params():
    """Test that init_state accepts the new parameters."""
    print("\nTesting init_state with custom cache parameters...")

    # Note: This is just a signature test - we won't actually run it
    # since it requires video frames
    from inspect import signature
    from efficient_track_anything.efficienttam_video_predictor import EfficientTAMVideoPredictor

    sig = signature(EfficientTAMVideoPredictor.init_state)
    params = list(sig.parameters.keys())

    expected_params = ['self', 'video_path', 'offload_video_to_cpu',
                       'offload_state_to_cpu', 'async_loading_frames',
                       'prefetch_count', 'cache_size', 'num_workers']

    for param in expected_params:
        if param in params:
            print(f"✓ Parameter '{param}' exists in init_state")
        else:
            print(f"✗ Parameter '{param}' missing from init_state")
            raise AssertionError(f"Missing parameter: {param}")

    print("\n✓ All init_state parameters present")


if __name__ == "__main__":
    test_build_with_params()
    test_init_state_params()
    print("\n" + "="*60)
    print("SUCCESS: All memory configuration parameters are working!")
    print("="*60)
