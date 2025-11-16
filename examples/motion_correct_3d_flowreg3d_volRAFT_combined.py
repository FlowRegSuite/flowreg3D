"""
Combined FlowReg3D + VolRAFT visualization script.

This script reproduces the standalone motion_correct_3d_test.py
and motion_correct_3d_volraft_test.py workflows, then opens three
napari viewers (moving vs fixed, FlowReg3D vs fixed, VolRAFT vs fixed)
with standard 50/50 red-green overlays. Each viewer automatically
captures 10 screenshots from different perspectives for zero-shot
inspection before handing control over to the user.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imageio
import napari
import numpy as np
import torch
from pyflowreg.util.io.factory import get_video_file_reader
from scipy.ndimage import gaussian_filter, zoom

from flowreg3d.core.optical_flow_3d import get_displacement as flowreg_get_displacement
from flowreg3d.core.optical_flow_3d import imregister_wrapper
from flowreg3d.motion_generation.motion_generators import (
    get_default_3d_generator,
    get_high_disp_3d_generator,
    get_low_disp_3d_generator,
    get_test_3d_generator,
)
from flowreg3d.util.random import fix_seed
from VolRAFT.models.model import ModelFactory
from VolRAFT.utils import CheckpointController, YAMLHandler

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

DEFAULT_VOLRAFT_CHECKPOINT_DIR = (
    Path(__file__).resolve().parents[1]
    / "VolRAFT"
    / "checkpoints"
    / "volraft_config_120"
    / "checkpoint_20240119_184617_980292"
)
VOLRAFT_CHECKPOINT_DIR = (
    Path(os.environ["VOLRAFT_CHECKPOINT_DIR"]).expanduser().resolve()
    if "VOLRAFT_CHECKPOINT_DIR" in os.environ
    else DEFAULT_VOLRAFT_CHECKPOINT_DIR
)

FLOWREG_MODE = os.environ.get("FLOWREG3D_MODE", "numba").lower()
VOLRAFT_MODE = os.environ.get("VOLRAFT_MODE", "volraft").lower()

SNAPSHOT_ROOT = Path(__file__).resolve().parents[1] / "results" / "napari_snapshots"
SNAPSHOTS_PER_COMBINATION = 10

CAMERA_PRESETS: Sequence[Dict[str, float]] = (
    {"name": "front", "elevation": 0.0, "azimuth": 0.0, "roll": 0.0, "zoom": 0.7},
    {"name": "oblique_pos", "elevation": 25.0, "azimuth": 35.0, "roll": 0.0, "zoom": 0.65},
    {"name": "oblique_neg", "elevation": -25.0, "azimuth": -35.0, "roll": 0.0, "zoom": 0.65},
    {"name": "top", "elevation": 90.0, "azimuth": 0.0, "roll": 0.0, "zoom": 0.75},
    {"name": "bottom", "elevation": -90.0, "azimuth": 0.0, "roll": 0.0, "zoom": 0.75},
    {"name": "left", "elevation": 0.0, "azimuth": 90.0, "roll": 0.0, "zoom": 0.7},
    {"name": "right", "elevation": 0.0, "azimuth": -90.0, "roll": 0.0, "zoom": 0.7},
    {"name": "roll_pos", "elevation": 20.0, "azimuth": 30.0, "roll": 20.0, "zoom": 0.6},
    {"name": "roll_neg", "elevation": -20.0, "azimuth": -30.0, "roll": -20.0, "zoom": 0.6},
    {"name": "tight_front", "elevation": 10.0, "azimuth": 0.0, "roll": 0.0, "zoom": 0.85},
)


# -------------------------------------------------------------------------
# Volume preparation & synthetic motion
# -------------------------------------------------------------------------

def process_3d_stack(video_data: np.ndarray) -> np.ndarray:
    """
    Resize, crop, and normalize a 3D stack.

    Args:
        video_data: Input array with shape (T, Y, X) or (T, Y, X, C).
    """
    print("\nProcessing 3D stack...")
    print(f"  Input shape: {video_data.shape}")

    video = video_data.astype(np.float32)

    resize_factor = 1.0
    print(f"  Resizing to {100 * resize_factor:.0f}%...")
    zoom_factors = (1.0, resize_factor, resize_factor) + ((1.0,) if video.ndim == 4 else ())
    video_resized = zoom(video, zoom_factors, order=1)
    print(f"  After resize: {video_resized.shape}")

    print("  Cropping 25px from XY boundaries...")
    if video_resized.shape[1] > 50 and video_resized.shape[2] > 50:
        slices = (slice(None), slice(25, -25), slice(25, -25))
        if video_resized.ndim == 4:
            slices += (slice(None),)
        video_cropped = video_resized[slices]
    else:
        print("  Video too small to crop; skipping")
        video_cropped = video_resized
    print(f"  After crop: {video_cropped.shape}")

    vmin = video_cropped.min()
    vmax = video_cropped.max()
    print(f"  Normalizing with range [{vmin:.4f}, {vmax:.4f}]")
    video_normalized = (video_cropped - vmin) / (vmax - vmin) if vmax > vmin else video_cropped

    print(f"  Final shape: {video_normalized.shape}")
    print(
        f"  Value range: [{video_normalized.min():.3f}, {video_normalized.max():.3f}]"
    )
    return video_normalized


def warp_volume_bw3d_torch(volume: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Backward warp via torch grid_sample (useful for experiments)."""
    flow = -flow
    v = torch.from_numpy(volume).float()
    f = torch.from_numpy(flow).float()
    if v.ndim == 3:
        v = v[..., None]
    Z, H, W, C = v.shape

    zz = torch.linspace(-1, 1, Z)
    yy = torch.linspace(-1, 1, H)
    xx = torch.linspace(-1, 1, W)
    grid = torch.meshgrid(zz, yy, xx, indexing="ij")
    nz = grid[0] + 2 * f[..., 0] / max(Z - 1, 1)
    ny = grid[1] + 2 * f[..., 1] / max(H - 1, 1)
    nx = grid[2] + 2 * f[..., 2] / max(W - 1, 1)
    sample_grid = torch.stack([nx, ny, nz], dim=-1)[None]

    v = v.permute(3, 0, 1, 2)[None]
    out = torch.nn.functional.grid_sample(
        v, sample_grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    out = out[0].permute(1, 2, 3, 0).numpy()
    return out[..., 0] if out.shape[-1] == 1 else out


def warp_volume_splat3d(volume: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Forward warp via scatter/splatting."""
    Z, H, W = volume.shape[:3]
    z, y, x = np.meshgrid(np.arange(Z), np.arange(H), np.arange(W), indexing="ij")

    tx = (x + flow[..., 0]).ravel()
    ty = (y + flow[..., 1]).ravel()
    tz = (z + flow[..., 2]).ravel()

    ix = np.floor(tx).astype(np.int64)
    fx = tx - ix
    iy = np.floor(ty).astype(np.int64)
    fy = ty - iy
    iz = np.floor(tz).astype(np.int64)
    fz = tz - iz

    ix0 = np.clip(ix, 0, W - 1)
    ix1 = np.clip(ix + 1, 0, W - 1)
    iy0 = np.clip(iy, 0, H - 1)
    iy1 = np.clip(iy + 1, 0, H - 1)
    iz0 = np.clip(iz, 0, Z - 1)
    iz1 = np.clip(iz + 1, 0, Z - 1)

    w000 = (1 - fx) * (1 - fy) * (1 - fz)
    w100 = fx * (1 - fy) * (1 - fz)
    w010 = (1 - fx) * fy * (1 - fz)
    w110 = fx * fy * (1 - fz)
    w001 = (1 - fx) * (1 - fy) * fz
    w101 = fx * (1 - fy) * fz
    w011 = (1 - fx) * fy * fz
    w111 = fx * fy * fz

    def splat(values: np.ndarray) -> np.ndarray:
        V = values.ravel()
        N = Z * H * W
        out = np.zeros(N, dtype=np.float64)
        den = np.zeros(N, dtype=np.float64)

        def idx(zz, yy, xx):
            return (zz * H + yy) * W + xx

        for w, zz, yy, xx in (
            (w000, iz0, iy0, ix0),
            (w100, iz0, iy0, ix1),
            (w010, iz0, iy1, ix0),
            (w110, iz0, iy1, ix1),
            (w001, iz1, iy0, ix0),
            (w101, iz1, iy0, ix1),
            (w011, iz1, iy1, ix0),
            (w111, iz1, iy1, ix1),
        ):
            np.add.at(out, idx(zz, yy, xx), V * w)
            np.add.at(den, idx(zz, yy, xx), w)

        den[den == 0] = 1.0
        return (out / den).reshape(Z, H, W).astype(values.dtype)

    if volume.ndim == 4:
        return np.stack([splat(volume[..., c]) for c in range(volume.shape[3])], axis=-1)
    return splat(volume)


def warp_volume_pc3d(volume: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Forward warp via scipy.interpolate.griddata (reference implementation)."""
    from scipy.interpolate import griddata

    Z, H, W = volume.shape[:3]
    grid_z, grid_y, grid_x = np.meshgrid(
        np.arange(Z, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    target_x = grid_x + flow[..., 0]
    target_y = grid_y + flow[..., 1]
    target_z = grid_z + flow[..., 2]

    if volume.ndim == 4:
        warped = np.zeros_like(volume)
        tp = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
        sp = np.column_stack([target_x.ravel(), target_y.ravel(), target_z.ravel()])
        for c in range(volume.shape[-1]):
            warped[..., c] = griddata(
                sp, volume[..., c].ravel(), tp, method="linear", fill_value=0
            ).reshape(Z, H, W)
        return warped

    tp = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    sp = np.column_stack([target_x.ravel(), target_y.ravel(), target_z.ravel()])
    return griddata(sp, volume.ravel(), tp, method="linear", fill_value=0).reshape(Z, H, W)


def create_displaced_frame_with_generator(
    video: np.ndarray, generator_type: str = "high_disp"
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply synthetic motion using the motion generator suite."""
    print("\nCreating displaced frame with synthetic motion...")
    print(f"  Generator type: {generator_type}")

    depth, height, width = video.shape[:3]
    if generator_type == "default":
        generator = get_default_3d_generator()
    elif generator_type == "low_disp":
        generator = get_low_disp_3d_generator()
    elif generator_type == "test":
        generator = get_test_3d_generator()
    elif generator_type == "high_disp":
        generator = get_high_disp_3d_generator()
    else:
        print("  Unknown generator type, falling back to high_disp")
        generator = get_high_disp_3d_generator()

    flow_gt, invalid_mask = generator(depth=depth, height=height, width=width)
    print(f"  Ground truth flow shape: {flow_gt.shape}")
    for axis, name in enumerate("xyz"):
        component = flow_gt[..., axis]
        print(
            f"    d{name}: min={component.min():.2f}, max={component.max():.2f}, mean={component.mean():.2f}"
        )

    displaced = warp_volume_splat3d(video, flow_gt)
    boundary = 10
    slices = tuple(slice(boundary, -boundary) for _ in range(3))
    displaced = displaced[slices + ((slice(None),) if displaced.ndim == 4 else ())]
    flow_gt = flow_gt[slices + (slice(None),)]
    if invalid_mask is not None:
        invalid_mask = invalid_mask[slices]
    print(f"  Warping complete, cropped {boundary}px boundaries → {displaced.shape}")
    return displaced, flow_gt


def evaluate_flow_accuracy(
    flow_est: np.ndarray, flow_gt: np.ndarray, boundary: int = 25
) -> float:
    """Compute End-Point Error excluding boundary voxels."""
    if boundary > 0:
        slices = tuple(slice(boundary, -boundary) for _ in range(3))
        flow_est = flow_est[slices + (slice(None),)]
        flow_gt = flow_gt[slices + (slice(None),)]
    epe = np.mean(np.linalg.norm(flow_est - flow_gt, axis=-1))
    return float(epe)


# -------------------------------------------------------------------------
# FlowReg3D solvers
# -------------------------------------------------------------------------

def compute_flowreg3d_flow(
    frame1: np.ndarray, frame2: np.ndarray, flow_params: Dict
) -> np.ndarray:
    """Compute FlowReg3D optical flow via the Numba backend."""
    print("\nComputing FlowReg3D optical flow (NumPy backend)...")
    f1 = frame1.astype(np.float32)
    f2 = frame2.astype(np.float32)
    if f1.ndim == 3:
        f1 = f1[..., None]
        f2 = f2[..., None]

    print("  Applying Gaussian smoothing (sigma=0.5)...")
    for c in range(f1.shape[-1]):
        f1[..., c] = gaussian_filter(f1[..., c], sigma=0.5)
        f2[..., c] = gaussian_filter(f2[..., c], sigma=0.5)

    mins = f1.min(axis=(0, 1, 2), keepdims=True)
    maxs = f1.max(axis=(0, 1, 2), keepdims=True)
    ranges = np.where(maxs - mins > 0, maxs - mins, 1.0)
    f1_norm = (f1 - mins) / ranges
    f2_norm = (f2 - mins) / ranges

    t0 = time.perf_counter()
    flow = flowreg_get_displacement(f1_norm, f2_norm, **flow_params)
    elapsed = time.perf_counter() - t0
    print(f"  Flow computed in {elapsed:.2f}s with FlowReg3D backend.")
    _print_flow_stats(flow)
    return flow


def compute_flowreg3d_flow_torch(
    frame1: np.ndarray, frame2: np.ndarray, flow_params: Dict
) -> np.ndarray:
    """Compute FlowReg3D optical flow via the PyTorch backend."""
    from flowreg3d.util.torch.image_processing_3D import (
        apply_gaussian_filter,
        normalize,
    )
    import flowreg3d.core.torch.optical_flow_3d as torch_of3d

    print("\nComputing FlowReg3D optical flow (PyTorch backend)...")
    t1 = torch.from_numpy(frame1).to(torch.float64)
    t2 = torch.from_numpy(frame2).to(torch.float64)
    if t1.ndim == 3:
        t1 = t1.unsqueeze(-1)
        t2 = t2.unsqueeze(-1)

    t1n = normalize(t1, ref=t1, channel_normalization="together")
    t2n = normalize(t2, ref=t1, channel_normalization="together")
    sigma_spatial = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
    t1n = apply_gaussian_filter(t1n, sigma_spatial)
    t2n = apply_gaussian_filter(t2n, sigma_spatial)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t1n = t1n.to(device)
    t2n = t2n.to(device)

    start = time.perf_counter()
    with torch.no_grad():
        flow = torch_of3d.get_displacement(t1n, t2n, **flow_params)
    elapsed = time.perf_counter() - start
    print(f"  Flow computed in {elapsed:.2f}s with FlowReg3D torch backend.")
    flow_np = flow.detach().cpu().numpy().astype(np.float64, copy=False)
    _print_flow_stats(flow_np)
    return flow_np


def _print_flow_stats(flow: np.ndarray) -> None:
    mags = np.sqrt(np.sum(flow**2, axis=-1))
    for axis, name in enumerate("xyz"):
        component = flow[..., axis]
        print(
            f"    d{name}: min={component.min():.2f}, max={component.max():.2f}, mean={component.mean():.2f}"
        )
    print(
        f"    |flow|: min={mags.min():.2f}, max={mags.max():.2f}, mean={mags.mean():.2f}"
    )


# -------------------------------------------------------------------------
# VolRAFT tiling + inference helpers
# -------------------------------------------------------------------------

def _build_foreground_mask(volume: np.ndarray, percentile: float = 10.0) -> np.ndarray:
    vol = volume.squeeze(-1)
    if vol.size == 0:
        return np.ones_like(vol, dtype=bool)
    threshold = np.percentile(vol, percentile)
    mask = vol > threshold
    if not mask.any():
        mask = vol > (vol.mean() if vol.ptp() > 0 else 0.0)
    return mask


def _gaussian_window_3d(shape: Tuple[int, int, int]) -> np.ndarray:
    dz, dy, dx = map(int, shape)
    mz, my, mx = [(s - 1.0) / 2.0 for s in (dz, dy, dx)]
    zz, yy, xx = np.ogrid[-mz : mz + 1, -my : my + 1, -mx : mx + 1]
    sigma = float(min(shape)) / 6.0 or 1.0
    window = np.exp(-(xx**2 + yy**2 + zz**2) / (2.0 * sigma**2))
    return window.astype(np.float32, copy=False)


def _window_starts(length: int, window: int, stride: int) -> np.ndarray:
    if length <= window:
        return np.array([0], dtype=int)
    starts = list(range(0, length - window + 1, stride))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return np.array(starts, dtype=int)


def _compute_patch_padding(
    volume_shape: Tuple[int, int, int],
    patch_shape: Tuple[int, int, int],
    margin_before: Tuple[int, int, int],
    margin_after: Tuple[int, int, int],
) -> Tuple[Tuple[int, int], ...]:
    padding = []
    for dim, patch_dim, before_margin, after_margin in zip(
        volume_shape, patch_shape, margin_before, margin_after
    ):
        before = before_margin
        after = after_margin
        total = dim + before + after
        if total < patch_dim:
            deficit = patch_dim - total
            extra_before = deficit // 2
            before += extra_before
            after += deficit - extra_before
        padding.append((before, after))
    return tuple(padding)


def run_volraft_inference(
    reference_frame: np.ndarray,
    moving_frame: np.ndarray,
    checkpoint_dir: Path,
    use_gpu: bool = True,
    num_overlaps: int = 5,
    mask_percentile: Optional[float] = 10.0,
) -> np.ndarray:
    """Run VolRAFT inference by tiling both volumes."""
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"VolRAFT checkpoint directory not found: {checkpoint_dir}. "
            "Set VOLRAFT_CHECKPOINT_DIR or place weights in the default location."
        )

    controller = CheckpointController(str(checkpoint_dir))
    config_path = controller.find_config_file()
    if config_path is None:
        raise FileNotFoundError(
            f"No configuration YAML found in {checkpoint_dir}. "
            "Ensure the VolRAFT checkpoints folder is intact."
        )

    config = YAMLHandler.read_yaml(config_path) or {}
    _, patch_shape, flow_shape, _, _, _, _, _ = controller.load_last_checkpoint(
        network=None, optimizer=None, scheduler=None
    )
    if patch_shape is None or flow_shape is None:
        raise RuntimeError("Checkpoint lacks patch/flow shapes required for inference.")

    model = ModelFactory.build_instance(
        patch_shape=patch_shape,
        flow_shape=flow_shape,
        config=config,
        ptdtype=torch.float32,
    )
    _, _, _, model, _, _, _, _ = controller.load_last_checkpoint(
        network=model, optimizer=None, scheduler=None
    )

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"  Using VolRAFT device: {device}")

    ref = reference_frame.astype(np.float32, copy=False)
    mov = moving_frame.astype(np.float32, copy=False)
    if ref.ndim == 3:
        ref = ref[..., None]
        mov = mov[..., None]
    if ref.shape[-1] > 1:
        print(f"  Averaging {ref.shape[-1]} channels for VolRAFT.")
        ref = ref.mean(axis=-1, keepdims=True)
        mov = mov.mean(axis=-1, keepdims=True)

    original_shape = reference_frame.shape[:3]
    patch_spatial = tuple(int(x) for x in patch_shape[-3:])
    flow_spatial = tuple(int(x) for x in flow_shape[-3:])
    margin_before = tuple(
        max((patch_dim - flow_dim) // 2, 0)
        for patch_dim, flow_dim in zip(patch_spatial, flow_spatial)
    )
    margin_after = tuple(
        max(patch_dim - flow_dim - before, 0)
        for patch_dim, flow_dim, before in zip(patch_spatial, flow_spatial, margin_before)
    )
    gaussian_window = _gaussian_window_3d(flow_spatial)
    default_full_margin = tuple(max(flow_dim // 4, 2) for flow_dim in flow_spatial)
    half_margin = tuple(min(m // 2, flow_dim // 2) for m, flow_dim in zip(default_full_margin, flow_spatial))
    core_slices = []
    for dim, hm in zip(flow_spatial, half_margin):
        start = hm
        end = dim - hm
        if end <= start:
            start = 0
            end = dim
        core_slices.append(slice(start, end))
    core_slices = tuple(core_slices)

    overlap_cap = max(1, min(int(num_overlaps), min(flow_spatial)))
    stride = tuple(max(dim // overlap_cap, 1) for dim in flow_spatial)
    print(
        f"  Tile stride set to {stride} voxels (~1/{overlap_cap} of flow size)."
    )

    mask = (
        np.ones(original_shape, dtype=bool)
        if mask_percentile is None
        else _build_foreground_mask(ref, percentile=float(mask_percentile))
    )

    padding = _compute_patch_padding(
        original_shape, patch_spatial, margin_before, margin_after
    )
    pad_spec = padding + ((0, 0),)
    if any(pad != (0, 0) for pad in padding):
        ref = np.pad(ref, pad_spec, mode="edge")
        mov = np.pad(mov, pad_spec, mode="edge")
        mask = np.pad(mask, padding, mode="constant", constant_values=False)
        print(f"  Applied padding {padding} to preserve context margins.")

    padded_shape = ref.shape[:3]
    weight_accum = np.zeros(padded_shape, dtype=np.float32)
    flow_accum = np.zeros(padded_shape + (3,), dtype=np.float32)
    pad_offsets = [pad[0] for pad in padding]

    gaussian_window = gaussian_window.astype(np.float32)
    mask = mask.astype(bool, copy=False)
    z_positions = _window_starts(original_shape[0], flow_spatial[0], stride[0])
    y_positions = _window_starts(original_shape[1], flow_spatial[1], stride[1])
    x_positions = _window_starts(original_shape[2], flow_spatial[2], stride[2])
    total_tiles = len(z_positions) * len(y_positions) * len(x_positions)
    print(
        f"  Running VolRAFT on {total_tiles} overlapping tiles "
        f"(patch {patch_spatial}, prediction window {flow_spatial})."
    )

    processed_tiles = 0
    for z0 in z_positions:
        z_flow_start = z0 + pad_offsets[0]
        z_flow_slice = slice(z_flow_start, z_flow_start + flow_spatial[0])
        z_patch_slice = slice(
            z_flow_start - margin_before[0],
            z_flow_start - margin_before[0] + patch_spatial[0],
        )
        for y0 in y_positions:
            y_flow_start = y0 + pad_offsets[1]
            y_flow_slice = slice(y_flow_start, y_flow_start + flow_spatial[1])
            y_patch_slice = slice(
                y_flow_start - margin_before[1],
                y_flow_start - margin_before[1] + patch_spatial[1],
            )
            for x0 in x_positions:
                x_flow_start = x0 + pad_offsets[2]
                x_flow_slice = slice(x_flow_start, x_flow_start + flow_spatial[2])
                x_patch_slice = slice(
                    x_flow_start - margin_before[2],
                    x_flow_start - margin_before[2] + patch_spatial[2],
                )

                patch_mask = mask[z_flow_slice, y_flow_slice, x_flow_slice]
                if not np.any(patch_mask):
                    continue

                ref_patch = np.ascontiguousarray(ref[z_patch_slice, y_patch_slice, x_patch_slice, :])
                mov_patch = np.ascontiguousarray(mov[z_patch_slice, y_patch_slice, x_patch_slice, :])
                ref_tensor = torch.from_numpy(np.moveaxis(ref_patch, -1, 0)[None]).to(device)
                mov_tensor = torch.from_numpy(np.moveaxis(mov_patch, -1, 0)[None]).to(device)

                with torch.no_grad():
                    flow_pred = model.forward(ref_tensor, mov_tensor)
                    if isinstance(flow_pred, list):
                        flow_pred = flow_pred[-1]

                flow_np_raw = flow_pred.squeeze(0).detach().cpu().numpy()
                flow_np_raw = np.moveaxis(flow_np_raw, 0, -1).astype(np.float32, copy=False)

                flow_np = np.empty_like(flow_np_raw)
                flow_np[..., 0] = flow_np_raw[..., 2]  # dx
                flow_np[..., 1] = flow_np_raw[..., 1]  # dy
                flow_np[..., 2] = flow_np_raw[..., 0]  # dz

                weights = np.zeros_like(patch_mask, dtype=np.float32)
                core_mask = patch_mask[core_slices]
                if not np.any(core_mask):
                    continue
                weights[core_slices] = gaussian_window[core_slices] * core_mask

                flow_weighted = flow_np * weights[..., None]
                flow_accum[z_flow_slice, y_flow_slice, x_flow_slice] += flow_weighted
                weight_accum[z_flow_slice, y_flow_slice, x_flow_slice] += weights
                processed_tiles += 1

    if processed_tiles == 0:
        raise RuntimeError("Foreground mask eliminated all patches; cannot run VolRAFT.")
    print(f"  Processed {processed_tiles} / {total_tiles} tiles.")

    valid = weight_accum > 0
    flow_accum[valid] /= weight_accum[valid][..., None]
    flow_accum[~valid] = 0

    if any(pad != (0, 0) for pad in padding):
        slices = tuple(slice(pad[0], pad[0] + orig) for pad, orig in zip(padding, original_shape))
        flow_accum = flow_accum[slices]
        mask = mask[slices]

    flow_accum *= mask[..., None]
    return flow_accum.astype(np.float32, copy=False)


# -------------------------------------------------------------------------
# Napari visualization + screenshot automation
# -------------------------------------------------------------------------

def _add_red_green_layers(
    viewer: napari.Viewer,
    fixed_volume: np.ndarray,
    moving_volume: np.ndarray,
    fixed_name: str,
    moving_name: str,
) -> None:
    viewer.dims.ndisplay = 3
    viewer.add_image(
        fixed_volume,
        name=fixed_name,
        colormap="green",
        blending="additive",
        opacity=0.5,
        contrast_limits=[0, 1],
        rendering="mip",
    )
    viewer.add_image(
        moving_volume,
        name=moving_name,
        colormap="red",
        blending="additive",
        opacity=0.5,
        contrast_limits=[0, 1],
        rendering="mip",
    )
    viewer.camera.center = tuple(dim / 2 for dim in fixed_volume.shape)


def capture_perspective_gallery(
    viewer: napari.Viewer,
    output_dir: Path,
    combination_name: str,
    volume_shape: Tuple[int, int, int],
    camera_presets: Sequence[Dict[str, float]] = CAMERA_PRESETS,
) -> None:
    """Capture SNAPSHOTS_PER_COMBINATION screenshots with unique viewpoints."""
    target_dir = output_dir / combination_name
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Capturing {SNAPSHOTS_PER_COMBINATION} screenshots in {target_dir}...")

    viewer.dims.ndisplay = 3
    viewer.dims.current_step = tuple(s // 2 for s in volume_shape)
    viewer.camera.center = tuple(dim / 2 for dim in volume_shape)

    for idx, preset in enumerate(camera_presets[:SNAPSHOTS_PER_COMBINATION], start=1):
        viewer.camera.elevation = preset["elevation"]
        viewer.camera.azimuth = preset["azimuth"]
        viewer.camera.roll = preset["roll"]
        viewer.camera.zoom = preset["zoom"]
        screenshot = viewer.screenshot(canvas_only=True)
        filename = target_dir / f"{combination_name}_{idx:02d}_{preset['name']}.png"
        imageio.imwrite(filename, screenshot)
        print(f"    • Saved {filename}")


# -------------------------------------------------------------------------
# Main workflow
# -------------------------------------------------------------------------

def main():
    """Run FlowReg3D + VolRAFT motion correction and visualization."""
    print("=" * 60)
    print("3D Motion Correction (FlowReg3D vs VolRAFT)")
    print("=" * 60)

    fix_seed(seed=1, deterministic=False, verbose=True)

    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    aligned_file = repo_root / "data" / "aligned_sequence" / "compensated.HDF5"
    if not aligned_file.exists():
        print(f"\n✗ Error: aligned file not found: {aligned_file}")
        print("  Please run the compensation pipeline first.")
        return None

    print(f"\nLoading pre-aligned video: {aligned_file}")
    reader = get_video_file_reader(str(aligned_file), buffer_size=100, bin_size=9)
    print(f"  Reader shape: {reader.shape}")
    print(f"  Frames: {reader.frame_count}")
    print(f"  Channels: {reader.n_channels}")

    video_3d = []
    while reader.has_batch():
        video_3d.append(reader.read_batch())
    video_3d = np.concatenate(video_3d, axis=0) if video_3d else reader[:]
    reader.close()

    print(f"\nLoaded 3D stack shape: {video_3d.shape}")
    processed = process_3d_stack(video_3d)
    displaced, flow_gt = create_displaced_frame_with_generator(processed, generator_type="high_disp")

    boundary = 10
    if processed.ndim == 4:
        original_cropped = processed[
            boundary:-boundary, boundary:-boundary, boundary:-boundary, :
        ]
    else:
        original_cropped = processed[
            boundary:-boundary, boundary:-boundary, boundary:-boundary
        ]

    print("\nPreparing FlowReg3D computation...")
    flowreg_params = {
        "alpha": (0.25, 0.25, 0.25),
        "iterations": 100,
        "a_data": 0.45,
        "a_smooth": 1.0,
        "weight": np.array([0.5, 0.5], dtype=np.float64),
        "levels": 50,
        "eta": 0.8,
        "update_lag": 5,
        "min_level": 5,
        "const_assumption": "gc",
        "uvw": None,
    }

    if FLOWREG_MODE == "torch":
        flowreg_flow = compute_flowreg3d_flow_torch(original_cropped, displaced, flowreg_params)
    else:
        flowreg_flow = compute_flowreg3d_flow(original_cropped, displaced, flowreg_params)

    print("\nApplying FlowReg3D motion correction...")
    flowreg_corrected = imregister_wrapper(
        displaced,
        flowreg_flow[..., 0],
        flowreg_flow[..., 1],
        flowreg_flow[..., 2],
        original_cropped,
        interpolation_method="cubic",
    )
    flowreg_epe = evaluate_flow_accuracy(flowreg_flow, flow_gt, boundary=25)
    diff_displaced = np.mean(np.abs(original_cropped - displaced))
    diff_corrected = np.mean(np.abs(original_cropped - flowreg_corrected))
    print(f"  FlowReg3D EPE: {flowreg_epe:.2f} pixels")
    print(f"  FlowReg3D MAD (original vs displaced): {diff_displaced:.4f}")
    print(f"  FlowReg3D MAD (original vs corrected): {diff_corrected:.4f}")
    print(f"  FlowReg3D improvement: {diff_displaced / diff_corrected:.2f}x")

    print("\nPreparing VolRAFT computation...")
    volraft_params = {
        "checkpoint_dir": VOLRAFT_CHECKPOINT_DIR,
        "use_gpu": True,
        "num_overlaps": 5,
        "mask_percentile": 10.0,
    }

    if VOLRAFT_MODE == "volraft":
        t0 = time.perf_counter()
        volraft_flow = -run_volraft_inference(
            original_cropped,
            displaced,
            checkpoint_dir=volraft_params["checkpoint_dir"],
            use_gpu=volraft_params["use_gpu"],
            num_overlaps=volraft_params["num_overlaps"],
            mask_percentile=volraft_params["mask_percentile"],
        )
        elapsed = time.perf_counter() - t0
        print(f"  VolRAFT flow computed in {elapsed:.2f}s.")
    elif VOLRAFT_MODE == "torch":
        volraft_flow = compute_flowreg3d_flow_torch(original_cropped, displaced, flowreg_params)
    else:
        raise ValueError(f"Unsupported VOLRAFT_MODE '{VOLRAFT_MODE}' (use 'volraft' or 'torch').")

    print("\nApplying VolRAFT motion correction...")
    volraft_corrected = imregister_wrapper(
        displaced,
        volraft_flow[..., 0],
        volraft_flow[..., 1],
        volraft_flow[..., 2],
        original_cropped,
        interpolation_method="cubic",
    )
    volraft_epe = evaluate_flow_accuracy(volraft_flow, flow_gt, boundary=25)
    volraft_diff = np.mean(np.abs(original_cropped - volraft_corrected))
    print(f"  VolRAFT EPE: {volraft_epe:.2f} pixels")
    print(f"  VolRAFT MAD (original vs corrected): {volraft_diff:.4f}")
    print(f"  VolRAFT improvement: {diff_displaced / volraft_diff:.2f}x")

    snapshot_root = SNAPSHOT_ROOT
    print(f"\nSaving napari screenshots under: {snapshot_root}")
    viewers: List[napari.Viewer] = []

    viewer_moving = napari.Viewer(title="Moving vs Fixed (Red/Green)")
    _add_red_green_layers(
        viewer_moving,
        fixed_volume=original_cropped,
        moving_volume=displaced,
        fixed_name="Fixed (Green)",
        moving_name="Moving (Red)",
    )
    capture_perspective_gallery(
        viewer_moving,
        snapshot_root,
        "moving_vs_fixed",
        original_cropped.shape[:3],
    )
    viewers.append(viewer_moving)

    viewer_flowreg = napari.Viewer(title="FlowReg3D Corrected vs Fixed")
    _add_red_green_layers(
        viewer_flowreg,
        fixed_volume=original_cropped,
        moving_volume=flowreg_corrected,
        fixed_name="Fixed (Green)",
        moving_name="FlowReg3D Corrected (Red)",
    )
    capture_perspective_gallery(
        viewer_flowreg,
        snapshot_root,
        "flowreg3d_vs_fixed",
        original_cropped.shape[:3],
    )
    viewers.append(viewer_flowreg)

    viewer_volraft = napari.Viewer(title="VolRAFT Corrected vs Fixed")
    _add_red_green_layers(
        viewer_volraft,
        fixed_volume=original_cropped,
        moving_volume=volraft_corrected,
        fixed_name="Fixed (Green)",
        moving_name="VolRAFT Corrected (Red)",
    )
    capture_perspective_gallery(
        viewer_volraft,
        snapshot_root,
        "volraft_vs_fixed",
        original_cropped.shape[:3],
    )
    viewers.append(viewer_volraft)

    print("\nAll viewers ready. Interact with napari windows or close them to exit.")
    napari.run()

    return {
        "original": original_cropped,
        "displaced": displaced,
        "flowreg_corrected": flowreg_corrected,
        "volraft_corrected": volraft_corrected,
        "flowreg_flow": flowreg_flow,
        "volraft_flow": volraft_flow,
        "flow_gt": flow_gt,
    }


if __name__ == "__main__":
    main()
