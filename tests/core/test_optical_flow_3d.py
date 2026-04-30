"""Tests for 3D optical-flow stage dispatch and GNC plumbing."""

import numpy as np
import pytest

import flowreg3d.core.optical_flow_3d as optical_flow_3d
from flowreg3d.core.level_solver_3d import compute_flow_3d_gnc


def _compress_runs(values):
    """Collapse consecutive duplicate values while preserving order."""
    compressed = []
    for value in values:
        if not compressed or compressed[-1] != value:
            compressed.append(value)
    return compressed


def _make_zero_level_solver(sink=None):
    """Create a fake 3D level solver that records GNC stages."""

    def fake_level_solver(*args):
        u = args[11]
        v = args[12]
        w = args[13]
        gnc_beta = args[-1]
        if sink is not None:
            sink.append(gnc_beta)
        return np.zeros_like(u), np.zeros_like(v), np.zeros_like(w)

    return fake_level_solver


def _make_tensor_stub(counter=None):
    """Return zero 3D motion tensors with the expected padded shape."""

    def fake_get_motion_tensor_gc(f1, f2, hz, hy, hx):
        if counter is not None:
            counter.append((f1.shape, f2.shape, hz, hy, hx))
        shape = (f1.shape[0] + 2, f1.shape[1] + 2, f1.shape[2] + 2)
        return tuple(np.zeros(shape, dtype=np.float64) for _ in range(10))

    return fake_get_motion_tensor_gc


def test_get_displacement_without_gnc_uses_default_stage(monkeypatch):
    """Test the default path does not pass a GNC stage to the solver."""
    fixed = np.zeros((12, 12, 12), dtype=np.float64)
    moving = np.zeros((12, 12, 12), dtype=np.float64)
    seen_betas = []

    def fake_level_solver(
        J11,
        J22,
        J33,
        J44,
        J12,
        J13,
        J23,
        J14,
        J24,
        J34,
        weight,
        u,
        v,
        w,
        alpha,
        iterations,
        update_lag,
        verbose,
        a_data,
        a_smooth,
        hx,
        hy,
        hz,
        gnc_beta=None,
    ):
        seen_betas.append(gnc_beta)
        return np.zeros_like(u), np.zeros_like(v), np.zeros_like(w)

    monkeypatch.setattr(optical_flow_3d, "level_solver", fake_level_solver)

    flow = optical_flow_3d.get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
    )

    assert flow.shape == (12, 12, 12, 3)
    assert _compress_runs(seen_betas) == [None]


def test_get_displacement_without_gnc_ignores_warping_steps(monkeypatch):
    """Test warping_steps alone does not change the default path."""
    fixed = np.zeros((12, 12, 12), dtype=np.float64)
    moving = np.zeros((12, 12, 12), dtype=np.float64)
    seen_betas = []

    def fake_level_solver(
        J11,
        J22,
        J33,
        J44,
        J12,
        J13,
        J23,
        J14,
        J24,
        J34,
        weight,
        u,
        v,
        w,
        alpha,
        iterations,
        update_lag,
        verbose,
        a_data,
        a_smooth,
        hx,
        hy,
        hz,
        gnc_beta=None,
    ):
        seen_betas.append(gnc_beta)
        return np.zeros_like(u), np.zeros_like(v), np.zeros_like(w)

    monkeypatch.setattr(optical_flow_3d, "level_solver", fake_level_solver)

    flow = optical_flow_3d.get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        warping_steps=3,
    )

    assert flow.shape == (12, 12, 12, 3)
    assert _compress_runs(seen_betas) == [None]


def test_get_displacement_with_gnc_warping_steps_repeats_solver_calls(monkeypatch):
    """Test GNC warping steps repeat the per-level solver calls."""
    fixed = np.zeros((12, 12, 12), dtype=np.float64)
    moving = np.zeros((12, 12, 12), dtype=np.float64)
    base_seen_betas = []
    warp_seen_betas = []

    def make_fake_level_solver(sink):
        def fake_level_solver(
            J11,
            J22,
            J33,
            J44,
            J12,
            J13,
            J23,
            J14,
            J24,
            J34,
            weight,
            u,
            v,
            w,
            alpha,
            iterations,
            update_lag,
            verbose,
            a_data,
            a_smooth,
            hx,
            hy,
            hz,
            gnc_beta=None,
        ):
            sink.append(gnc_beta)
            return np.zeros_like(u), np.zeros_like(v), np.zeros_like(w)

        return fake_level_solver

    monkeypatch.setattr(
        optical_flow_3d, "level_solver", make_fake_level_solver(base_seen_betas)
    )
    optical_flow_3d.get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        gnc_schedule=(0.0, 0.5, 1.0),
        warping_steps=1,
    )

    monkeypatch.setattr(
        optical_flow_3d, "level_solver", make_fake_level_solver(warp_seen_betas)
    )
    optical_flow_3d.get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        gnc_schedule=(0.0, 0.5, 1.0),
        warping_steps=3,
    )

    assert _compress_runs(base_seen_betas) == [0.0, 0.5, 1.0]
    for beta in (0.0, 0.5, 1.0):
        assert warp_seen_betas.count(beta) == 3 * base_seen_betas.count(beta)


def test_get_displacement_with_gnc_default_warping_steps_is_ten(monkeypatch):
    """Test omitted GNC warping steps use Sun-style ten warps per level."""
    fixed = np.zeros((12, 12, 12), dtype=np.float64)
    moving = np.zeros((12, 12, 12), dtype=np.float64)
    base_seen_betas = []
    seen_betas = []

    monkeypatch.setattr(
        optical_flow_3d,
        "level_solver",
        _make_zero_level_solver(base_seen_betas),
    )
    optical_flow_3d.get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        gnc_schedule=(0.0, 1.0),
        warping_steps=1,
    )

    monkeypatch.setattr(
        optical_flow_3d,
        "level_solver",
        _make_zero_level_solver(seen_betas),
    )
    flow = optical_flow_3d.get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        gnc_schedule=(0.0, 1.0),
    )

    assert flow.shape == (12, 12, 12, 3)
    assert seen_betas.count(0.0) == 10 * base_seen_betas.count(0.0)
    assert seen_betas.count(1.0) == 10 * base_seen_betas.count(1.0)


def test_get_displacement_with_gnc_rebuilds_tensors_each_warp(monkeypatch):
    """Test each GNC warp rewarps volumes and rebuilds motion tensors."""
    fixed = np.zeros((12, 12, 12, 2), dtype=np.float64)
    moving = np.zeros((12, 12, 12, 2), dtype=np.float64)
    warp_calls = []
    tensor_calls = []
    seen_betas = []

    def fake_imregister_wrapper(f2_level, u, v, w, f1_level):
        warp_calls.append((f2_level.shape, u.shape, v.shape, w.shape, f1_level.shape))
        return f2_level

    monkeypatch.setattr(optical_flow_3d, "imregister_wrapper", fake_imregister_wrapper)
    monkeypatch.setattr(
        optical_flow_3d,
        "get_motion_tensor_gc",
        _make_tensor_stub(tensor_calls),
    )
    monkeypatch.setattr(
        optical_flow_3d,
        "level_solver",
        _make_zero_level_solver(seen_betas),
    )

    flow = optical_flow_3d.get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        gnc_schedule=(0.0, 1.0),
        warping_steps=2,
    )

    assert flow.shape == (12, 12, 12, 3)
    assert len(warp_calls) == len(seen_betas)
    assert len(tensor_calls) == 2 * len(warp_calls)


def test_get_displacement_with_gnc_median_filters_after_each_warp(monkeypatch):
    """Test GNC applies the median filter after each warp-level solve."""
    fixed = np.zeros((12, 12, 12), dtype=np.float64)
    moving = np.zeros((12, 12, 12), dtype=np.float64)
    median_calls = []
    seen_betas = []

    def fake_median_filter(arr, size, mode):
        median_calls.append((arr.shape, size, mode))
        return arr

    monkeypatch.setattr(optical_flow_3d, "median_filter", fake_median_filter)
    monkeypatch.setattr(optical_flow_3d, "get_motion_tensor_gc", _make_tensor_stub())
    monkeypatch.setattr(
        optical_flow_3d,
        "level_solver",
        _make_zero_level_solver(seen_betas),
    )

    flow = optical_flow_3d.get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        gnc_schedule=(0.0, 1.0),
        warping_steps=3,
    )

    assert flow.shape == (12, 12, 12, 3)
    assert len(median_calls) == 3 * len(seen_betas)
    assert {(size, mode) for _, size, mode in median_calls} == {((5, 5, 5), "mirror")}
    assert all(min(shape) > 5 for shape, _, _ in median_calls)


def test_get_displacement_with_gnc_carries_flow_between_stages(monkeypatch):
    """Test each GNC stage starts from the previous stage displacement."""
    fixed = np.zeros((12, 12, 12), dtype=np.float64)
    moving = np.zeros((12, 12, 12), dtype=np.float64)
    stage_initial_means = []

    def fake_level_solver(*args):
        u = args[11]
        v = args[12]
        w = args[13]
        gnc_beta = args[-1]
        stage_initial_means.append((gnc_beta, float(u[1:-1, 1:-1, 1:-1].mean())))
        du = np.full_like(u, gnc_beta + 1.0)
        return du, np.zeros_like(v), np.zeros_like(w)

    monkeypatch.setattr(optical_flow_3d, "level_solver", fake_level_solver)

    flow = optical_flow_3d.get_displacement(
        fixed,
        moving,
        levels=1,
        iterations=2,
        update_lag=1,
        gnc_schedule=(0.0, 1.0),
        warping_steps=1,
    )

    assert flow.shape == (12, 12, 12, 3)
    beta0_means = [mean for beta, mean in stage_initial_means if beta == 0.0]
    beta1_means = [mean for beta, mean in stage_initial_means if beta == 1.0]
    assert beta0_means[0] == 0.0
    assert beta0_means[-1] == float(len(beta0_means) - 1)
    assert beta1_means[0] == float(len(beta0_means))


def test_level_solver_dispatches_to_default_solver_without_gnc(monkeypatch):
    """Test the CPU wrapper keeps the default solver branch separate."""
    shape = (3, 3, 3, 1)
    field_shape = shape[:3]
    seen = []

    def fake_default(*args, **kwargs):
        seen.append(("default", kwargs))
        return np.zeros((*field_shape, 3), dtype=np.float64)

    def fake_gnc(*args, **kwargs):
        seen.append(("gnc", kwargs))
        return np.zeros((*field_shape, 3), dtype=np.float64)

    monkeypatch.setattr(optical_flow_3d, "compute_flow_3d", fake_default)
    monkeypatch.setattr(optical_flow_3d, "compute_flow_3d_gnc", fake_gnc)

    optical_flow_3d.level_solver(
        *[np.zeros(shape, dtype=np.float64) for _ in range(10)],
        np.ones(shape, dtype=np.float64),
        np.zeros(field_shape, dtype=np.float64),
        np.zeros(field_shape, dtype=np.float64),
        np.zeros(field_shape, dtype=np.float64),
        (1.0, 1.0, 1.0),
        1,
        1,
        0,
        np.array([0.45], dtype=np.float64),
        1.0,
        1.0,
        1.0,
        1.0,
    )

    assert len(seen) == 1
    assert seen[0][0] == "default"
    assert "gnc_beta" not in seen[0][1]


def test_level_solver_dispatches_to_gnc_solver_with_beta(monkeypatch):
    """Test the CPU wrapper only enters the GNC solver when requested."""
    shape = (3, 3, 3, 1)
    field_shape = shape[:3]
    seen = []

    def fake_default(*args, **kwargs):
        seen.append(("default", kwargs))
        return np.zeros((*field_shape, 3), dtype=np.float64)

    def fake_gnc(*args, **kwargs):
        seen.append(("gnc", kwargs))
        return np.zeros((*field_shape, 3), dtype=np.float64)

    monkeypatch.setattr(optical_flow_3d, "compute_flow_3d", fake_default)
    monkeypatch.setattr(optical_flow_3d, "compute_flow_3d_gnc", fake_gnc)

    optical_flow_3d.level_solver(
        *[np.zeros(shape, dtype=np.float64) for _ in range(10)],
        np.ones(shape, dtype=np.float64),
        np.zeros(field_shape, dtype=np.float64),
        np.zeros(field_shape, dtype=np.float64),
        np.zeros(field_shape, dtype=np.float64),
        (1.0, 1.0, 1.0),
        1,
        1,
        0,
        np.array([0.45], dtype=np.float64),
        1.0,
        1.0,
        1.0,
        1.0,
        gnc_beta=0.5,
    )

    assert len(seen) == 1
    assert seen[0][0] == "gnc"
    assert seen[0][1]["gnc_beta"] == 0.5


def test_compute_flow_3d_gnc_updates_weights_on_matlab_cadence():
    """Test GNC weights update on the 2D/MATLAB-style update-lag tick."""
    shape = (3, 3, 3, 1)
    field_shape = shape[:3]
    tensors = [np.zeros(shape, dtype=np.float64) for _ in range(10)]
    J11, J22, J33, J44, J12, J13, J23, J14, J24, J34 = tensors
    J11[1, 1, 1, 0] = 1.0
    J14[1, 1, 1, 0] = -1.0
    J44[1, 1, 1, 0] = 4.0
    weight = np.ones(shape, dtype=np.float64)
    zeros = np.zeros(field_shape, dtype=np.float64)
    common = dict(
        J11=J11,
        J22=J22,
        J33=J33,
        J44=J44,
        J12=J12,
        J13=J13,
        J23=J23,
        J14=J14,
        J24=J24,
        J34=J34,
        weight=weight,
        u=zeros,
        v=zeros,
        w=zeros,
        alpha_x=1.0,
        alpha_y=1.0,
        alpha_z=1.0,
        iterations=1,
        a_data=np.array([0.45], dtype=np.float64),
        a_smooth=1.0,
        hx=1.0,
        hy=1.0,
        hz=1.0,
    )

    lagged_quadratic = compute_flow_3d_gnc(**common, update_lag=2, gnc_beta=0.0)
    lagged_robust = compute_flow_3d_gnc(**common, update_lag=2, gnc_beta=1.0)
    np.testing.assert_allclose(lagged_quadratic, lagged_robust)

    updated_quadratic = compute_flow_3d_gnc(**common, update_lag=1, gnc_beta=0.0)
    updated_midpoint = compute_flow_3d_gnc(**common, update_lag=1, gnc_beta=0.5)
    updated_robust = compute_flow_3d_gnc(**common, update_lag=1, gnc_beta=1.0)

    assert updated_quadratic[1, 1, 1, 0] > updated_midpoint[1, 1, 1, 0]
    assert updated_midpoint[1, 1, 1, 0] > updated_robust[1, 1, 1, 0]


def test_compute_flow_3d_gnc_leaves_quadratic_data_term_beta_invariant():
    """Test ``a_data=1`` is quadratic already and ignores the GNC beta."""
    shape = (3, 3, 3, 1)
    field_shape = shape[:3]
    tensors = [np.zeros(shape, dtype=np.float64) for _ in range(10)]
    J11, J22, J33, J44, J12, J13, J23, J14, J24, J34 = tensors
    J11[1, 1, 1, 0] = 1.0
    J14[1, 1, 1, 0] = -1.0
    J44[1, 1, 1, 0] = 4.0
    zeros = np.zeros(field_shape, dtype=np.float64)
    common = dict(
        J11=J11,
        J22=J22,
        J33=J33,
        J44=J44,
        J12=J12,
        J13=J13,
        J23=J23,
        J14=J14,
        J24=J24,
        J34=J34,
        weight=np.ones(shape, dtype=np.float64),
        u=zeros,
        v=zeros,
        w=zeros,
        alpha_x=1.0,
        alpha_y=1.0,
        alpha_z=1.0,
        iterations=1,
        update_lag=1,
        a_data=np.array([1.0], dtype=np.float64),
        a_smooth=1.0,
        hx=1.0,
        hy=1.0,
        hz=1.0,
    )

    quadratic_stage = compute_flow_3d_gnc(**common, gnc_beta=0.0)
    robust_stage = compute_flow_3d_gnc(**common, gnc_beta=1.0)

    np.testing.assert_allclose(quadratic_stage, robust_stage)


def test_compute_flow_3d_gnc_blends_sublinear_smoothness_weights():
    """Test sublinear smoothness weights change with the GNC stage beta."""
    common = _smoothness_case_3d(a_smooth=0.5)

    quadratic_stage = compute_flow_3d_gnc(**common, update_lag=1, gnc_beta=0.0)
    robust_stage = compute_flow_3d_gnc(**common, update_lag=1, gnc_beta=1.0)

    assert not np.allclose(quadratic_stage, robust_stage)


def test_compute_flow_3d_gnc_leaves_quadratic_smoothness_beta_invariant():
    """Test ``a_smooth=1`` is quadratic already and ignores the GNC beta."""
    common = _smoothness_case_3d(a_smooth=1.0)

    quadratic_stage = compute_flow_3d_gnc(**common, update_lag=1, gnc_beta=0.0)
    robust_stage = compute_flow_3d_gnc(**common, update_lag=1, gnc_beta=1.0)

    np.testing.assert_allclose(quadratic_stage, robust_stage)


def _smoothness_case_3d(a_smooth):
    """Create a non-flat 3D flow so smoothness weights affect the solve."""
    shape = (5, 5, 5, 1)
    field_shape = shape[:3]
    u = np.zeros(field_shape, dtype=np.float64)
    u[2, 2, 2] = 1.0
    return dict(
        J11=np.zeros(shape, dtype=np.float64),
        J22=np.zeros(shape, dtype=np.float64),
        J33=np.zeros(shape, dtype=np.float64),
        J44=np.zeros(shape, dtype=np.float64),
        J12=np.zeros(shape, dtype=np.float64),
        J13=np.zeros(shape, dtype=np.float64),
        J23=np.zeros(shape, dtype=np.float64),
        J14=np.zeros(shape, dtype=np.float64),
        J24=np.zeros(shape, dtype=np.float64),
        J34=np.zeros(shape, dtype=np.float64),
        weight=np.ones(shape, dtype=np.float64),
        u=u,
        v=np.zeros(field_shape, dtype=np.float64),
        w=np.zeros(field_shape, dtype=np.float64),
        alpha_x=1.0,
        alpha_y=1.0,
        alpha_z=1.0,
        iterations=1,
        a_data=np.array([1.0], dtype=np.float64),
        a_smooth=a_smooth,
        hx=1.0,
        hy=1.0,
        hz=1.0,
    )


def test_torch_level_solver_dispatch_keeps_default_branch_separate(monkeypatch):
    """Test torch default calls do not enter the GNC-capable solver."""
    torch = pytest.importorskip("torch")
    import flowreg3d.core.torch.optical_flow_3d as torch_optical_flow_3d

    shape = (3, 3, 3, 1)
    field_shape = shape[:3]
    seen = []

    def fake_default(*args, **kwargs):
        seen.append(("default", kwargs))
        return (
            torch.zeros(field_shape, dtype=torch.float64),
            torch.zeros(field_shape, dtype=torch.float64),
            torch.zeros(field_shape, dtype=torch.float64),
        )

    def fake_gnc(*args, **kwargs):
        seen.append(("gnc", kwargs))
        return (
            torch.zeros(field_shape, dtype=torch.float64),
            torch.zeros(field_shape, dtype=torch.float64),
            torch.zeros(field_shape, dtype=torch.float64),
        )

    monkeypatch.setattr(
        torch_optical_flow_3d, "level_solver_rbgs3d_torch", fake_default
    )
    monkeypatch.setattr(
        torch_optical_flow_3d, "level_solver_rbgs3d_torch_gnc", fake_gnc
    )

    torch_optical_flow_3d.level_solver(
        *[torch.zeros(shape, dtype=torch.float64) for _ in range(10)],
        torch.ones(shape, dtype=torch.float64),
        torch.zeros(field_shape, dtype=torch.float64),
        torch.zeros(field_shape, dtype=torch.float64),
        torch.zeros(field_shape, dtype=torch.float64),
        (1.0, 1.0, 1.0),
        1,
        1,
        None,
        torch.tensor([0.45], dtype=torch.float64),
        1.0,
        1.0,
        1.0,
        1.0,
    )

    assert seen == [("default", {})]


def test_torch_level_solver_dispatch_uses_gnc_branch_with_beta(monkeypatch):
    """Test torch GNC calls use the separate GNC-capable solver."""
    torch = pytest.importorskip("torch")
    import flowreg3d.core.torch.optical_flow_3d as torch_optical_flow_3d

    shape = (3, 3, 3, 1)
    field_shape = shape[:3]
    seen = []

    def fake_default(*args, **kwargs):
        seen.append(("default", kwargs))
        return (
            torch.zeros(field_shape, dtype=torch.float64),
            torch.zeros(field_shape, dtype=torch.float64),
            torch.zeros(field_shape, dtype=torch.float64),
        )

    def fake_gnc(*args, **kwargs):
        seen.append(("gnc", kwargs))
        return (
            torch.zeros(field_shape, dtype=torch.float64),
            torch.zeros(field_shape, dtype=torch.float64),
            torch.zeros(field_shape, dtype=torch.float64),
        )

    monkeypatch.setattr(
        torch_optical_flow_3d, "level_solver_rbgs3d_torch", fake_default
    )
    monkeypatch.setattr(
        torch_optical_flow_3d, "level_solver_rbgs3d_torch_gnc", fake_gnc
    )

    torch_optical_flow_3d.level_solver(
        *[torch.zeros(shape, dtype=torch.float64) for _ in range(10)],
        torch.ones(shape, dtype=torch.float64),
        torch.zeros(field_shape, dtype=torch.float64),
        torch.zeros(field_shape, dtype=torch.float64),
        torch.zeros(field_shape, dtype=torch.float64),
        (1.0, 1.0, 1.0),
        1,
        1,
        None,
        torch.tensor([0.45], dtype=torch.float64),
        1.0,
        1.0,
        1.0,
        1.0,
        gnc_beta=0.5,
    )

    assert len(seen) == 1
    assert seen[0][0] == "gnc"
    assert seen[0][1]["update_lag_semantics"] == "matlab"
    assert seen[0][1]["gnc_beta"] == 0.5
