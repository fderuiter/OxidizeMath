//! Snapshot pixel rendering, aerodynamic diagnostic readouts, and fairing plot overlays
//! for [`super::LbmUnified`] and [`super::super::lbm_fairing_ui::LbmUiExtension`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::lbm_fairing::{LbmFairingMetrics, LBM_CORE_RADIUS, LBM_CORE_X, LBM_CORE_Y};
use super::LbmSnapshotMetrics;
use crate::async_sim::StateSnapshot;
use crate::framework::CalloutManager;
use eframe::egui::{self, Color32};
use egui_plot::{Points, Polygon};
use math_commons::math_kernel::colormap::{
    ensure_dark_canvas_contrast, heatmap_color, heatmap_color_high_contrast,
};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    BgkCollision, LatticeBoltzmannD2Q9,
};

pub(crate) fn inspect_solver_solid_and_core(
    solver: &LatticeBoltzmannD2Q9<BgkCollision>,
) -> (usize, bool) {
    let mut n_solid = 0_usize;
    let mut core_intact = true;
    for y in 0..solver.height() {
        for x in 0..solver.width() {
            let obs = solver.is_obstacle(x, y);
            if obs {
                n_solid += 1;
            }
            let dx = x as i32 - LBM_CORE_X;
            let dy = y as i32 - LBM_CORE_Y;
            if dx * dx + dy * dy <= LBM_CORE_RADIUS * LBM_CORE_RADIUS && !obs {
                core_intact = false;
            }
        }
    }
    (n_solid, core_intact)
}

pub(crate) fn render_lbm_snapshot_buffers(
    solver: &LatticeBoltzmannD2Q9<BgkCollision>,
    vorticity_buf: &[f64],
    display_vorticity: bool,
    high_contrast: bool,
) -> (Vec<Color32>, Vec<f64>, f64) {
    let width = solver.width();
    let height = solver.height();
    let mut pixels = vec![Color32::BLACK; width * height];
    let mut scalar_values = vec![0.0_f64; width * height];
    let mut max_vort = 0.0_f64;
    let obs_color = if high_contrast {
        let rgb = ensure_dark_canvas_contrast([180, 180, 180]);
        Color32::from_rgb(rgb[0], rgb[1], rgb[2])
    } else {
        Color32::from_gray(50)
    };
    for y in 0..height {
        let solver_y = height - 1 - y;
        for x in 0..width {
            let pixel_idx = flatten_2d_index(x, y, width);
            let solver_idx = flatten_2d_index(x, solver_y, width);
            if solver.is_obstacle(x, solver_y) {
                pixels[pixel_idx] = obs_color;
                scalar_values[pixel_idx] = -0.01;
            } else {
                let omega = vorticity_buf[solver_idx];
                let safe_w = if omega.is_finite() { omega } else { 0.0 };
                max_vort = max_vort.max(safe_w.abs());
                let norm = if display_vorticity {
                    scalar_values[pixel_idx] = safe_w;
                    (0.5 + safe_w * 12.0).clamp(0.0, 1.0)
                } else {
                    let v = solver.get_velocity_magnitude(x, solver_y);
                    let safe_v = if v.is_finite() { v } else { 0.0 };
                    scalar_values[pixel_idx] = safe_v;
                    (safe_v * 1000.0 / 255.0).clamp(0.0, 1.0)
                };
                let rgb = if high_contrast {
                    heatmap_color_high_contrast(norm)
                } else {
                    heatmap_color(norm)
                };
                pixels[pixel_idx] = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
            }
        }
    }
    (pixels, scalar_values, max_vort)
}

pub(crate) fn draw_aerodynamic_metrics(ui: &mut egui::Ui, snapshot: Option<&StateSnapshot>) {
    let Some(metrics) = snapshot
        .and_then(|s| s.structured_data.as_ref())
        .and_then(|d| d.downcast_ref::<LbmSnapshotMetrics>())
    else {
        return;
    };
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    ui.separator();
    ui.label("Aerodynamic Diagnostics");
    let ld_ratio = if metrics.drag_coeff.abs() > 1e-6 {
        metrics.lift_coeff / metrics.drag_coeff
    } else {
        0.0
    };
    if let Some(s) = snapshot.filter(|s| s.custom_data.len() >= 3) {
        ui.small(format!(
            "Forces F_D: {:+.4}   F_L: {:+.4}   C_D(core): {:+.3}",
            s.custom_data[0], s.custom_data[1], s.custom_data[2]
        ));
    }
    ui.colored_label(
        palette.info,
        format!(
            "Lift C_L: {:+.3}   Drag C_D: {:+.3}",
            metrics.lift_coeff, metrics.drag_coeff
        ),
    );
    ui.colored_label(
        palette.muted_text,
        format!(
            "L/D Ratio: {:+.2}   Max |ω_z|: {:.4}",
            ld_ratio, metrics.max_vorticity
        ),
    );
}

pub(crate) fn draw_fairing_canvas_overlay(
    visuals: &egui::Visuals,
    plot_ui: &mut egui_plot::PlotUi<'_>,
    cursor_x: i32,
    cursor_y: i32,
) {
    let palette = crate::app::SemanticPalette::from_visuals(visuals);
    let warn_rgb = ensure_dark_canvas_contrast([
        palette.warning.r(),
        palette.warning.g(),
        palette.warning.b(),
    ]);
    let core_color = Color32::from_rgb(warn_rgb[0], warn_rgb[1], warn_rgb[2]);
    let ring_pts: Vec<[f64; 2]> = (0..24)
        .map(|k| {
            let theta = std::f64::consts::TAU * (k as f64) / 24.0;
            [
                LBM_CORE_X as f64 + 0.5 + (LBM_CORE_RADIUS as f64) * theta.cos(),
                LBM_CORE_Y as f64 + 0.5 + (LBM_CORE_RADIUS as f64) * theta.sin(),
            ]
        })
        .collect();
    plot_ui.polygon(
        Polygon::new("Protected Core (R=4)", ring_pts)
            .fill_color(Color32::TRANSPARENT)
            .stroke(egui::Stroke::new(1.5_f32, core_color)),
    );
    plot_ui.points(
        Points::new(
            "Sculpt Cursor",
            vec![[cursor_x as f64 + 0.5, cursor_y as f64 + 0.5]],
        )
        .radius(4.0_f32)
        .color(palette.accent),
    );
}

pub(crate) fn sync_fairing_callouts(m: &LbmFairingMetrics, callouts: &mut CalloutManager) {
    let core_pos = [LBM_CORE_X as f64 + 0.5, LBM_CORE_Y as f64 + 0.5];
    let core_lbl = format!("Core R=4 | N={}", m.n_solid);
    callouts.set_auto_callout("lbm_core", core_pos, core_lbl, [-36.0, -26.0]);
    let drag_pos = [LBM_CORE_X as f64 + 14.0, LBM_CORE_Y as f64 + 10.0];
    let drag_lbl = format!("C_D = {:.3} (≤ {:.3})", m.c_d, 0.82 * m.c_d_cylinder);
    callouts.set_auto_callout("lbm_drag", drag_pos, drag_lbl, [28.0, -16.0]);
}
