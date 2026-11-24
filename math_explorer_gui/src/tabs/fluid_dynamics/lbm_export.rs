//! Snapshot construction and 2D scalar-field extraction for [`LbmUnified`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::lattice_boltzmann::render::{inspect_solver_solid_and_core, render_lbm_snapshot_pixels};
use super::lattice_boltzmann::{LbmSnapshotMetrics, LbmUnified};
use super::lbm_fairing::{LBM_REF_DIAMETER, LBM_RHO_0};
use crate::async_sim::StateSnapshot;
use crate::framework::ScalarField2D;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::compute_drag_coefficient;
use std::sync::{Arc, RwLock};

fn sample_lbm_scalar_and_mask(model: &LbmUnified) -> (Vec<f64>, Vec<bool>) {
    let width = model.solver.width();
    let height = model.solver.height();
    let cell_count = width * height;
    let mut scalar_field = vec![0.0_f64; cell_count];
    let mut active_mask = vec![true; cell_count];

    for y in 0..height {
        let solver_y = height - 1 - y;
        for x in 0..width {
            let pixel_idx = flatten_2d_index(x, y, width);
            let solver_idx = flatten_2d_index(x, solver_y, width);
            if model.solver.is_obstacle(x, solver_y) {
                active_mask[pixel_idx] = false;
            } else if model.display_vorticity {
                let omega = model.vorticity_buf[solver_idx];
                scalar_field[pixel_idx] = if omega.is_finite() { omega } else { 0.0 };
            } else {
                let v = model.solver.get_velocity_magnitude(x, solver_y);
                scalar_field[pixel_idx] = if v.is_finite() { v } else { 0.0 };
            }
        }
    }

    (scalar_field, active_mask)
}

/// Builds a [`StateSnapshot`] for [`LbmUnified`] including top-down hydrodynamic scalar samples
/// and an obstacle-aware active fluid mask for 2D vector contour / heightmap / normal-map export.
pub(super) fn build_lbm_snapshot(model: &LbmUnified) -> StateSnapshot {
    let width = model.solver.width();
    let height = model.solver.height();
    let (pixels, max_vort) = render_lbm_snapshot_pixels(
        &model.solver,
        &model.vorticity_buf,
        model.display_vorticity,
        model.high_contrast,
    );
    let (scalar_field, active_mask) = sample_lbm_scalar_and_mask(model);
    let (f_drag, f_lift) = model.solver.hydrodynamic_forces();
    let c_d = compute_drag_coefficient(
        f_drag,
        LBM_RHO_0,
        model.inlet_velocity.max(1e-4),
        LBM_REF_DIAMETER,
    );
    let (n_solid, core_intact) = inspect_solver_solid_and_core(&model.solver);
    let metrics = LbmSnapshotMetrics {
        lift_coeff: model.latest_aero.lift_coeff,
        drag_coeff: model.latest_aero.drag_coeff,
        max_vorticity: max_vort,
        scalar_field,
        active_mask,
    };

    StateSnapshot {
        width,
        height,
        pixels: Arc::new(RwLock::new(pixels)),
        custom_data: vec![
            f_drag,
            f_lift,
            c_d,
            n_solid as f64,
            f64::from(u8::from(core_intact)),
            model.step_count as f64,
        ],
        structured_data: Some(Box::new(metrics)),
        sonification: crate::audio::SonificationFrame::silent(),
    }
}

/// Extracts a 2D continuous hydrodynamic scalar field with obstacle masking from `snapshot`.
pub(super) fn extract_lbm_scalar_field(snapshot: &StateSnapshot) -> Option<ScalarField2D> {
    if snapshot.width < 2 || snapshot.height < 2 {
        return None;
    }
    let metrics = snapshot
        .structured_data
        .as_ref()?
        .downcast_ref::<LbmSnapshotMetrics>()?;
    let expected = snapshot.width * snapshot.height;
    if metrics.scalar_field.len() != expected || metrics.active_mask.len() != expected {
        return None;
    }
    Some(
        ScalarField2D::continuous(
            snapshot.width,
            snapshot.height,
            metrics.scalar_field.clone(),
            false,
        )
        .with_active_mask(metrics.active_mask.clone()),
    )
}
