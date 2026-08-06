import inspect
import sys

import numpy as np
import pytest
import torch

import flowreg3d.core.torch.optical_flow_3d as torch_optical_flow
from flowreg3d.core.optical_flow_3d import imregister_wrapper as scipy_warp


def _zero_displacements(shape, dtype=torch.float64):
    zero = torch.zeros(shape, dtype=dtype)
    return zero.clone(), zero.clone(), zero.clone()


def test_cubic_warp_is_bit_exact_with_scipy_reference():
    pytest.importorskip("interpol")
    generator = np.random.default_rng(20260805)
    shape = (9, 10, 11, 2)
    moving = generator.normal(size=shape)
    fixed = generator.normal(size=shape)
    u = generator.normal(scale=0.7, size=shape[:3])
    v = generator.normal(scale=0.7, size=shape[:3])
    w = generator.normal(scale=0.7, size=shape[:3])

    expected = scipy_warp(moving, u, v, w, fixed, "cubic")
    tensors = [
        torch.as_tensor(value, dtype=torch.float64)
        for value in (moving, u, v, w, fixed)
    ]
    observed = torch_optical_flow.imregister_wrapper(
        *tensors, interpolation_method="cubic"
    ).numpy()

    assert observed.dtype == np.float32
    assert np.array_equal(observed, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cubic_warp_cuda_is_bit_exact_with_scipy_reference():
    pytest.importorskip("interpol")
    generator = np.random.default_rng(20260806)
    shape = (5, 6, 7, 2)
    moving = generator.normal(size=shape)
    fixed = generator.normal(size=shape)
    u = generator.normal(scale=0.5, size=shape[:3])
    v = generator.normal(scale=0.5, size=shape[:3])
    w = generator.normal(scale=0.5, size=shape[:3])

    expected = scipy_warp(moving, u, v, w, fixed, "cubic")
    tensors = [
        torch.as_tensor(value, dtype=torch.float64, device="cuda")
        for value in (moving, u, v, w, fixed)
    ]
    observed = (
        torch_optical_flow.imregister_wrapper(*tensors, interpolation_method="cubic")
        .cpu()
        .numpy()
    )

    assert np.array_equal(observed, expected)


def test_cubic_warp_single_channel_shape_and_out_of_bounds_replacement():
    pytest.importorskip("interpol")
    shape = (5, 6, 7)
    moving = torch.arange(np.prod(shape), dtype=torch.float64).reshape(shape)
    fixed = torch.full(shape, -123.25, dtype=torch.float64)
    u, v, w = _zero_displacements(shape)
    u[0, 0, 0] = -1.0
    u[-1, -1, -1] = 1.0

    observed = torch_optical_flow.imregister_wrapper(
        moving, u, v, w, fixed, interpolation_method="cubic"
    )

    assert observed.shape == shape
    assert observed.dtype == torch.float32
    assert observed[0, 0, 0] == fixed[0, 0, 0].to(torch.float32)
    assert observed[-1, -1, -1] == fixed[-1, -1, -1].to(torch.float32)

    in_bounds = torch.ones(shape, dtype=torch.bool)
    in_bounds[0, 0, 0] = False
    in_bounds[-1, -1, -1] = False
    assert torch.equal(observed[in_bounds], moving.to(torch.float32)[in_bounds])


@pytest.mark.parametrize("interpolation_method", ["bilinear", "nearest"])
def test_grid_sample_interpolation_routing_and_out_of_bounds_replacement(
    monkeypatch, interpolation_method
):
    shape = (4, 5, 6)
    moving = torch.arange(np.prod(shape) * 2, dtype=torch.float64).reshape(*shape, 2)
    fixed = torch.full_like(moving, -37.0)
    u, v, w = _zero_displacements(shape)
    u.fill_(0.4)
    u[0, 0, 0] = -1.0

    observed_modes = []
    grid_sample = torch_optical_flow.F.grid_sample

    def recording_grid_sample(*args, **kwargs):
        observed_modes.append(kwargs["mode"])
        return grid_sample(*args, **kwargs)

    monkeypatch.setattr(torch_optical_flow.F, "grid_sample", recording_grid_sample)
    observed = torch_optical_flow.imregister_wrapper(
        moving,
        u,
        v,
        w,
        fixed,
        interpolation_method=interpolation_method,
    )

    assert observed_modes == [interpolation_method]
    assert observed.shape == (*shape, 2)
    assert observed.dtype == moving.dtype
    assert torch.equal(observed[0, 0, 0], fixed[0, 0, 0])


def test_invalid_interpolation_method_raises_value_error():
    shape = (3, 4, 5)
    moving = torch.zeros(shape)
    fixed = torch.zeros_like(moving)
    u, v, w = _zero_displacements(shape, dtype=moving.dtype)

    with pytest.raises(ValueError, match="Unsupported interpolation method"):
        torch_optical_flow.imregister_wrapper(
            moving, u, v, w, fixed, interpolation_method="lanczos"
        )

    with pytest.raises(ValueError, match="Unsupported interpolation method"):
        torch_optical_flow.get_displacement(
            moving, fixed, interpolation_method="lanczos"
        )


def test_missing_cubic_dependency_has_actionable_error(monkeypatch):
    shape = (3, 4, 5)
    moving = torch.zeros(shape)
    fixed = torch.zeros_like(moving)
    u, v, w = _zero_displacements(shape, dtype=moving.dtype)
    monkeypatch.setitem(sys.modules, "interpol", None)

    with pytest.raises(ImportError, match=r"flowreg3d\[gpu\]"):
        torch_optical_flow.imregister_wrapper(
            moving, u, v, w, fixed, interpolation_method="cubic"
        )


@pytest.mark.parametrize(
    ("interpolation_kwargs", "expected_method"),
    [({}, "cubic"), ({"interpolation_method": "nearest"}, "nearest")],
)
def test_get_displacement_propagates_interpolation_method(
    monkeypatch, interpolation_kwargs, expected_method
):
    observed_methods = []

    def recording_warp(
        f2_level,
        u,
        v,
        w,
        f1_level,
        interpolation_method,
    ):
        del u, v, w, f1_level
        observed_methods.append(interpolation_method)
        return f2_level

    def zero_level_solver(*args, **kwargs):
        del kwargs
        u, v, w = args[11:14]
        return torch.zeros_like(u), torch.zeros_like(v), torch.zeros_like(w)

    monkeypatch.setattr(torch_optical_flow, "imregister_wrapper", recording_warp)
    monkeypatch.setattr(torch_optical_flow, "level_solver", zero_level_solver)

    volume = torch.zeros((10, 10, 10), dtype=torch.float32)
    torch_optical_flow.get_displacement(
        volume,
        volume,
        levels=1,
        iterations=1,
        **interpolation_kwargs,
    )

    assert observed_methods == [expected_method]


def test_get_displacement_interpolation_method_is_keyword_only():
    parameter = inspect.signature(torch_optical_flow.get_displacement).parameters[
        "interpolation_method"
    ]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == "cubic"
