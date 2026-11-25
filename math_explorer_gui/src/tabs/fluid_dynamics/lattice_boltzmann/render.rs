//! Snapshot pixel rendering, aerodynamic diagnostic readouts, and fairing plot overlays
//! for [`super::LbmUnified`] and [`super::super::lbm_fairing_ui::LbmUiExtension`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::lbm_fairing::{LbmFairingMetrics, LBM_CORE_RADIUS, LBM_CORE_X, LBM_CORE_Y};
use super::{LbmSnapshotMetrics, LBM_HEIGHT, LBM_WIDTH};
use crate::accessibility::draw_legend_swatch;
use crate::app::{contrast_ratio, SemanticPalette, ThemeMode};
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
use std::collections::HashMap;

/// Parses a `"ThemeMode:*"` or legacy `"HighContrast:*"` worker command into a [`ThemeMode`].
#[must_use]
pub(super) fn parse_lbm_theme_command(action: &str) -> Option<ThemeMode> {
    if let Some(rest) = action.strip_prefix("ThemeMode:") {
        Some(match rest {
            "Light" => ThemeMode::Light,
            "HighContrast" => ThemeMode::HighContrast,
            _ => ThemeMode::Dark,
        })
    } else if action.starts_with("HighContrast:") {
        Some(if action.ends_with('1') {
            ThemeMode::HighContrast
        } else {
            ThemeMode::Dark
        })
    } else {
        None
    }
}

/// Returns `true` when `color` belongs to the active `theme`'s rimmed/hatched LBM obstacle palette.
#[cfg(test)]
#[inline]
#[must_use]
pub(crate) fn is_lbm_obstacle_pixel(color: Color32, theme: ThemeMode) -> bool {
    let p = theme.palette();
    color == p.obstacle_dark || color == p.obstacle_light
}

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

fn pick_obstacle_pixel_color(
    solver: &LatticeBoltzmannD2Q9<BgkCollision>,
    pixels: &[Color32],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    palette: &SemanticPalette,
) -> Color32 {
    let mut min_dark = f64::INFINITY;
    let mut min_light = f64::INFINITY;
    let mut has_fluid_neighbor = false;
    for (dx, dy) in [(-1_isize, 0_isize), (1, 0), (0, -1), (0, 1)] {
        let (nx, ny) = (x as isize + dx, y as isize + dy);
        if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
            let (ux, uy) = (nx as usize, ny as usize);
            if !solver.is_obstacle(ux, height - 1 - uy) {
                has_fluid_neighbor = true;
                let fc = pixels[flatten_2d_index(ux, uy, width)];
                min_dark = min_dark.min(contrast_ratio(palette.obstacle_dark, fc));
                min_light = min_light.min(contrast_ratio(palette.obstacle_light, fc));
            }
        }
    }
    if has_fluid_neighbor {
        if min_dark >= min_light {
            palette.obstacle_dark
        } else {
            palette.obstacle_light
        }
    } else {
        palette.obstacle_interior_hatch(x, y)
    }
}

pub(crate) fn render_lbm_snapshot_pixels(
    solver: &LatticeBoltzmannD2Q9<BgkCollision>,
    vorticity_buf: &[f64],
    display_vorticity: bool,
    theme: ThemeMode,
) -> (Vec<Color32>, f64) {
    let width = solver.width();
    let height = solver.height();
    let palette = theme.palette();
    let high_contrast = theme == ThemeMode::HighContrast;
    let mut pixels = vec![palette.surface_fill; width * height];
    let mut max_vort = 0.0_f64;

    for y in 0..height {
        let solver_y = height - 1 - y;
        for x in 0..width {
            if !solver.is_obstacle(x, solver_y) {
                let pixel_idx = flatten_2d_index(x, y, width);
                let solver_idx = flatten_2d_index(x, solver_y, width);
                let omega = vorticity_buf[solver_idx];
                let safe_w = if omega.is_finite() { omega } else { 0.0 };
                max_vort = max_vort.max(safe_w.abs());
                let norm = if display_vorticity {
                    (0.5 + safe_w * 12.0).clamp(0.0, 1.0)
                } else {
                    let v = solver.get_velocity_magnitude(x, solver_y);
                    (if v.is_finite() { v } else { 0.0 } * 1000.0 / 255.0).clamp(0.0, 1.0)
                };
                let rgb = if high_contrast {
                    heatmap_color_high_contrast(norm)
                } else {
                    heatmap_color(norm)
                };
                pixels[pixel_idx] = SemanticPalette::colormap_to_color32(rgb);
            }
        }
    }

    for y in 0..height {
        let solver_y = height - 1 - y;
        for x in 0..width {
            if solver.is_obstacle(x, solver_y) {
                let pixel_idx = flatten_2d_index(x, y, width);
                pixels[pixel_idx] =
                    pick_obstacle_pixel_color(solver, &pixels, x, y, width, height, &palette);
            }
        }
    }
    (pixels, max_vort)
}

pub(crate) fn build_lbm_a11y_summary(
    snapshot: Option<&StateSnapshot>,
    params: &HashMap<String, f64>,
    theme: ThemeMode,
) -> String {
    let field_label = if *params.get("field_view").unwrap_or(&0.0) >= 0.5 {
        "Vorticity ω_z"
    } else {
        "Velocity |u|"
    };
    let n_solid = snapshot
        .and_then(|s| s.custom_data.get(3))
        .map_or(0_usize, |&v| v.max(0.0).round() as usize);
    let metrics = snapshot
        .and_then(|s| s.structured_data.as_ref())
        .and_then(|d| d.downcast_ref::<LbmSnapshotMetrics>())
        .copied()
        .unwrap_or_default();
    format!(
        "Wind tunnel {LBM_WIDTH}x{LBM_HEIGHT} ({field_label}), Solid cells: {n_solid}, Lift C_L: {:+.3}, Drag C_D: {:+.3}, Max |ω_z|: {:.4}; Color/shape key: blue-cyan-yellow-red fluid ramp; solid obstacles rendered with high-contrast rim and diagonal [▨] hatch ({theme:?})",
        metrics.lift_coeff, metrics.drag_coeff, metrics.max_vorticity
    )
}

pub(crate) fn draw_lbm_legend(ui: &mut egui::Ui, theme: ThemeMode) {
    let palette = theme.palette();
    let hc = theme == ThemeMode::HighContrast;
    let low_rgb = if hc {
        heatmap_color_high_contrast(0.0)
    } else {
        heatmap_color(0.0)
    };
    let high_rgb = if hc {
        heatmap_color_high_contrast(1.0)
    } else {
        heatmap_color(1.0)
    };
    ui.horizontal_wrapped(|ui| {
        draw_legend_swatch(
            ui,
            SemanticPalette::colormap_to_color32(low_rgb),
            None,
            "Low Fluid",
        );
        draw_legend_swatch(
            ui,
            SemanticPalette::colormap_to_color32(high_rgb),
            None,
            "High Fluid",
        );
        draw_legend_swatch(
            ui,
            palette.obstacle_dark,
            Some("▨"),
            "Solid Obstacle [▨ Hatched + Rim]",
        );
    });
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
    let core_color = SemanticPalette::colormap_to_color32(warn_rgb);
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
