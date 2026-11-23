//! Cold-path snapshot field capture and [`crate::framework::ExportTable`] builder for [`super::lattice_boltzmann::LbmUnified`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::lattice_boltzmann::LbmSnapshotMetrics;
use crate::async_sim::StateSnapshot;
use crate::framework::{compute_matrix_stride, ExportCell, ExportTable};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    AerodynamicCoefficients, BgkCollision, LatticeBoltzmannD2Q9,
};

pub(super) fn capture_lbm_snapshot_metrics(
    solver: &LatticeBoltzmannD2Q9<BgkCollision>,
    aero: AerodynamicCoefficients,
    max_vorticity: f64,
) -> LbmSnapshotMetrics {
    let width = solver.width();
    let height = solver.height();
    LbmSnapshotMetrics {
        lift_coeff: aero.lift_coeff,
        drag_coeff: aero.drag_coeff,
        max_vorticity,
        density: solver.state.density().to_vec(),
        velocity_x: solver.state.velocity_x().to_vec(),
        velocity_y: solver.state.velocity_y().to_vec(),
        obstacles: (0..height)
            .flat_map(|y| (0..width).map(move |x| solver.is_obstacle(x, y)))
            .collect(),
    }
}

pub(super) fn build_lbm_export_tables(snapshot: &StateSnapshot) -> Vec<ExportTable> {
    let Some(metrics) = snapshot
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<LbmSnapshotMetrics>())
    else {
        return Vec::new();
    };
    let (width, height) = (snapshot.width, snapshot.height);
    let expected = width * height;
    if expected == 0
        || metrics.density.len() != expected
        || metrics.velocity_x.len() != expected
        || metrics.velocity_y.len() != expected
        || metrics.obstacles.len() != expected
    {
        return Vec::new();
    }
    let (r_stride, c_stride) = compute_matrix_stride(height, width);
    let mut rows = Vec::new();
    for y in (0..height).step_by(r_stride) {
        for x in (0..width).step_by(c_stride) {
            let idx = flatten_2d_index(x, y, width);
            let vx = metrics.velocity_x[idx];
            let vy = metrics.velocity_y[idx];
            rows.push(vec![
                ExportCell::UInt(y as u64),
                ExportCell::UInt(x as u64),
                ExportCell::Float(metrics.density[idx]),
                ExportCell::Float(vx),
                ExportCell::Float(vy),
                ExportCell::Float(vx.hypot(vy)),
                ExportCell::Bool(metrics.obstacles[idx]),
            ]);
        }
    }
    vec![ExportTable::new(
        "Fluid Lattice",
        vec![
            "row".into(),
            "col".into(),
            "density".into(),
            "velocity_x".into(),
            "velocity_y".into(),
            "speed".into(),
            "is_obstacle".into(),
        ],
        rows,
    )]
}
