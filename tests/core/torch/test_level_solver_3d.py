import numpy as np
import torch

from flowreg3d.core.level_solver_3d import compute_flow_3d
from flowreg3d.core.torch.level_solver_3d import level_solver_rbgs3d_torch


def _set_boundary_3d(array):
    array[:, 0, :] = array[:, 1, :]
    array[:, -1, :] = array[:, -2, :]
    array[:, :, 0] = array[:, :, 1]
    array[:, :, -1] = array[:, :, -2]
    array[0, :, :] = array[1, :, :]
    array[-1, :, :] = array[-2, :, :]


def _smooth_numerator(field, increment, k, j, i, ax, ay, az):
    center = field[k, j, i]
    return (
        ax * (field[k, j, i - 1] + increment[k, j, i - 1] - center)
        + ax * (field[k, j, i + 1] + increment[k, j, i + 1] - center)
        + ay * (field[k, j - 1, i] + increment[k, j - 1, i] - center)
        + ay * (field[k, j + 1, i] + increment[k, j + 1, i] - center)
        + az * (field[k - 1, j, i] + increment[k - 1, j, i] - center)
        + az * (field[k + 1, j, i] + increment[k + 1, j, i] - center)
    )


def _reference_single_iteration(
    tensors,
    weight,
    fields,
    alpha,
    spacings,
    omega,
):
    (
        J11,
        J22,
        J33,
        _J44,
        J12,
        J13,
        J23,
        J14,
        J24,
        J34,
    ) = tensors
    u, v, w = fields
    depth, height, width, _ = J11.shape
    du = np.zeros_like(u)
    dv = np.zeros_like(v)
    dw = np.zeros_like(w)
    ax = alpha[0] / (spacings[0] * spacings[0])
    ay = alpha[1] / (spacings[1] * spacings[1])
    az = alpha[2] / (spacings[2] * spacings[2])
    smooth_denominator = 2 * (ax + ay + az)

    def coordinates(color):
        for k in range(1, depth - 1):
            for j in range(1, height - 1):
                for i in range(1, width - 1):
                    if (k + j + i) % 2 == color:
                        yield k, j, i

    for color in (0, 1):
        for k, j, i in coordinates(color):
            numerator = _smooth_numerator(u, du, k, j, i, ax, ay, az)
            numerator -= np.sum(
                weight[k, j, i]
                * (
                    J14[k, j, i]
                    + J12[k, j, i] * dv[k, j, i]
                    + J13[k, j, i] * dw[k, j, i]
                )
            )
            denominator = smooth_denominator + np.sum(weight[k, j, i] * J11[k, j, i])
            target = numerator / denominator if denominator != 0 else 0.0
            du[k, j, i] = (1 - omega) * du[k, j, i] + omega * target

        for k, j, i in coordinates(color):
            numerator = _smooth_numerator(v, dv, k, j, i, ax, ay, az)
            numerator -= np.sum(
                weight[k, j, i]
                * (
                    J24[k, j, i]
                    + J12[k, j, i] * du[k, j, i]
                    + J23[k, j, i] * dw[k, j, i]
                )
            )
            denominator = smooth_denominator + np.sum(weight[k, j, i] * J22[k, j, i])
            target = numerator / denominator if denominator != 0 else 0.0
            dv[k, j, i] = (1 - omega) * dv[k, j, i] + omega * target

        for k, j, i in coordinates(color):
            numerator = _smooth_numerator(w, dw, k, j, i, ax, ay, az)
            numerator -= np.sum(
                weight[k, j, i]
                * (
                    J34[k, j, i]
                    + J13[k, j, i] * du[k, j, i]
                    + J23[k, j, i] * dv[k, j, i]
                )
            )
            denominator = smooth_denominator + np.sum(weight[k, j, i] * J33[k, j, i])
            target = numerator / denominator if denominator != 0 else 0.0
            dw[k, j, i] = (1 - omega) * dw[k, j, i] + omega * target

        _set_boundary_3d(du)
        _set_boundary_3d(dv)
        _set_boundary_3d(dw)

    return du, dv, dw


def test_coupled_system_matches_cpu_component_sequential_updates():
    shape = (6, 7, 8, 1)
    spatial_shape = shape[:3]
    ones = np.ones(shape, dtype=np.float64)
    zeros = np.zeros(shape, dtype=np.float64)
    zero_field = np.zeros(spatial_shape, dtype=np.float64)
    numpy_arguments = (
        2.0 * ones,
        2.0 * ones,
        2.0 * ones,
        zeros,
        0.9 * ones,
        0.9 * ones,
        0.9 * ones,
        -0.4 * ones,
        0.3 * ones,
        -0.2 * ones,
        ones,
        zero_field,
        zero_field.copy(),
        zero_field.copy(),
    )
    alpha = (0.0, 0.0, 0.0)
    iterations = 20
    update_lag = 5

    cpu_flow = compute_flow_3d(
        *numpy_arguments,
        *alpha,
        iterations,
        update_lag,
        np.ones(1, dtype=np.float64),
        1.0,
        1.0,
        1.0,
        1.0,
    )
    torch_arguments = tuple(torch.as_tensor(value) for value in numpy_arguments)
    torch_fields = level_solver_rbgs3d_torch(
        *torch_arguments,
        alpha,
        iterations,
        update_lag,
        torch.ones(1, dtype=torch.float64),
        1.0,
        1.0,
        1.0,
        1.0,
    )
    torch_flow = torch.stack(torch_fields, dim=-1).numpy()

    assert np.isfinite(torch_flow).all()
    core = (slice(1, -1), slice(1, -1), slice(1, -1))
    core_indices = np.indices(tuple(size - 2 for size in spatial_shape)) + 1
    red_mask = core_indices.sum(axis=0) % 2 == 0
    for color_mask in (red_mask, ~red_mask):
        np.testing.assert_allclose(
            torch_flow[core][color_mask],
            cpu_flow[core][color_mask],
            rtol=0.0,
            atol=1e-10,
        )


def test_single_iteration_matches_red_black_reference_and_refreshes_boundaries():
    rng = np.random.default_rng(20260806)
    shape = (5, 6, 7, 2)
    spatial_shape = shape[:3]
    channel_scale = np.array([1.0, 1.15], dtype=np.float64).reshape(1, 1, 1, 2)
    tensors = (
        np.broadcast_to(2.5 * channel_scale, shape).copy(),
        np.broadcast_to(3.0 * channel_scale, shape).copy(),
        np.broadcast_to(3.5 * channel_scale, shape).copy(),
        np.ones(shape, dtype=np.float64),
        np.broadcast_to(0.35 * channel_scale, shape).copy(),
        np.broadcast_to(-0.25 * channel_scale, shape).copy(),
        np.broadcast_to(0.20 * channel_scale, shape).copy(),
        rng.normal(-0.3, 0.08, size=shape),
        rng.normal(0.2, 0.08, size=shape),
        rng.normal(-0.1, 0.08, size=shape),
    )
    weight = rng.uniform(0.2, 1.0, size=shape)
    weight /= weight.sum(axis=-1, keepdims=True)
    z, y, x = np.indices(spatial_shape, dtype=np.float64)
    fields = (
        0.03 * x - 0.01 * y,
        -0.02 * x + 0.015 * z,
        0.01 * y - 0.025 * z,
    )
    alpha = (0.3, 0.4, 0.5)
    spacings = (0.8, 1.1, 1.3)
    omega = 1.2
    expected = _reference_single_iteration(
        tensors, weight, fields, alpha, spacings, omega
    )

    observed = level_solver_rbgs3d_torch(
        *(torch.from_numpy(value) for value in tensors),
        torch.from_numpy(weight),
        *(torch.from_numpy(value) for value in fields),
        alpha,
        1,
        1,
        torch.ones(shape[-1], dtype=torch.float64),
        1.0,
        *spacings,
        omega=omega,
    )

    for actual, reference in zip(observed, expected):
        np.testing.assert_allclose(actual.numpy(), reference, rtol=0.0, atol=1e-12)
        assert torch.equal(actual[:, 0, :], actual[:, 1, :])
        assert torch.equal(actual[:, -1, :], actual[:, -2, :])
        assert torch.equal(actual[:, :, 0], actual[:, :, 1])
        assert torch.equal(actual[:, :, -1], actual[:, :, -2])
        assert torch.equal(actual[0, :, :], actual[1, :, :])
        assert torch.equal(actual[-1, :, :], actual[-2, :, :])


def test_zero_denominators_produce_finite_zero_updates():
    shape = (4, 5, 6, 1)
    zeros = torch.zeros(shape, dtype=torch.float64)
    linear_terms = (
        torch.full(shape, 1.0, dtype=torch.float64),
        torch.full(shape, -2.0, dtype=torch.float64),
        torch.full(shape, 3.0, dtype=torch.float64),
    )
    field = torch.zeros(shape[:3], dtype=torch.float64)

    result = level_solver_rbgs3d_torch(
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        *linear_terms,
        1.0,
        field,
        field,
        field,
        (0.0, 0.0, 0.0),
        2,
        1,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )

    for component in result:
        assert torch.isfinite(component).all()
        assert torch.count_nonzero(component) == 0


def test_nonlinear_solver_smoke_preserves_dtype_and_device():
    generator = torch.Generator(device="cpu").manual_seed(20260806)
    shape = (5, 6, 7, 2)
    spatial_shape = shape[:3]
    dtype = torch.float32
    ones = torch.ones(shape, dtype=dtype)
    weight = torch.rand(shape, generator=generator, dtype=dtype) + 0.1
    weight /= weight.sum(dim=-1, keepdim=True)
    fields = tuple(
        0.05 * torch.randn(spatial_shape, generator=generator, dtype=dtype)
        for _ in range(3)
    )

    result = level_solver_rbgs3d_torch(
        2.2 * ones,
        2.7 * ones,
        3.1 * ones,
        ones,
        0.2 * ones,
        -0.15 * ones,
        0.1 * ones,
        -0.3 * ones,
        0.2 * ones,
        -0.1 * ones,
        weight,
        *fields,
        (0.4, 0.5, 0.6),
        4,
        2,
        torch.tensor([0.45, 0.7], dtype=dtype),
        0.5,
        0.9,
        1.1,
        1.3,
    )

    for component in result:
        assert component.shape == spatial_shape
        assert component.dtype == dtype
        assert component.device.type == "cpu"
        assert not component.requires_grad
        assert torch.isfinite(component).all()
        assert torch.count_nonzero(component[1:-1, 1:-1, 1:-1]) > 0
