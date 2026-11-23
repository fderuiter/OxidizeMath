#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! 2D phase-portrait canvas, bounded quiver vector field (`egui_plot::Arrows`),
//! marching-squares nullclines, click-to-seed orbits, and synchronized $(x(t), y(t))$ twin timelines.

use super::comparison::{simulate_ode, OdeSimSpec, SolverTrajectory};
use super::recovery;
use super::systems::build_ode_param_map;
use super::OdeSolverTool;
use crate::framework::recovery::FiniteStateGuard;
use eframe::egui;
use egui_plot::{Arrows, Legend, Line, LineStyle, Plot, PlotPoints, Points};

/// Hard ceiling on quiver grid resolution ($25 \times 25 = 625$ arrows).
pub const MAX_QUIVER_GRID: usize = 25;

/// Default quiver grid resolution per axis.
pub const DEFAULT_QUIVER_GRID: usize = 17;

/// Hard ceiling on user click-seeded phase-plane orbits.
pub const MAX_SEEDED_ORBITS: usize = 8;

/// Default half-width of the symmetric 2D phase-plane domain window.
pub const DEFAULT_PHASE_EXTENT: f64 = 4.0;

/// Default $x' = f(t, x, y)$ formula for `OdePreset::PlanarCustom`.
pub const DEFAULT_PLANAR_X_EXPR: &str = "y";

/// Default $y' = g(t, x, y)$ formula for `OdePreset::PlanarCustom` (Van der Pol limit cycle).
pub const DEFAULT_PLANAR_Y_EXPR: &str = "k * (1 - x^2) * y - x";

/// Curated coupled 2D nonlinear ODE presets `(Title, x' formula, y' formula)`.
pub const PLANAR_COUPLED_PRESETS: &[(&str, &str, &str)] = &[
    ("Van der Pol Limit Cycle", "y", "k * (1 - x^2) * y - x"),
    (
        "Lotka-Volterra Predator-Prey",
        "r * x - k * x * y",
        "k * x * y - y",
    ),
    ("Damped Nonlinear Pendulum", "y", "-k * sin(x) - 0.25 * y"),
    ("Duffing Double-Well", "y", "x - x^3 - 0.3 * k * y"),
    (
        "Non-Autonomous Forced Spiral",
        "y - 0.25 * x",
        "-k * x + 0.4 * cos(t)",
    ),
];

/// Line segment `[[x0, y0], [x1, y1]]` along a planar nullcline contour.
pub type NullclineSegment = [[f64; 2]; 2];

fn march_cell_zero_segment(
    x0: f64,
    y0: f64,
    step: f64,
    v00: f64,
    v10: f64,
    v11: f64,
    v01: f64,
) -> Option<NullclineSegment> {
    if ![v00, v10, v11, v01].iter().all(|v| v.is_finite()) {
        return None;
    }
    let interp = |va: f64, vb: f64| -> Option<f64> {
        if (va <= 0.0 && vb > 0.0) || (va >= 0.0 && vb < 0.0) {
            let denom = va - vb;
            (denom.abs() > 1e-15).then(|| (va / denom).clamp(0.0, 1.0))
        } else {
            None
        }
    };
    let mut hits = [[0.0_f64; 2]; 4];
    let mut count = 0_usize;
    for (edge, hit) in [
        (interp(v00, v10), [x0, y0]),
        (interp(v10, v11), [x0 + step, y0]),
        (interp(v01, v11), [x0, y0 + step]),
        (interp(v00, v01), [x0, y0]),
    ]
    .into_iter()
    .enumerate()
    {
        if let Some(s) = hit.0 {
            hits[count] = if edge % 2 == 0 {
                [hit.1[0] + s * step, hit.1[1]]
            } else {
                [hit.1[0], hit.1[1] + s * step]
            };
            count += 1;
        }
    }
    (count >= 2).then_some([hits[0], hits[1]])
}

impl OdeSolverTool {
    /// Returns `true` when 2D phase-portrait rendering is enabled and supported by the active preset.
    #[must_use]
    pub fn is_phase_plane_active(&self) -> bool {
        self.phase_plane_mode && self.has_second_state()
    }

    /// Computes a deterministic symmetric domain half-width enclosing active and seeded initial states.
    #[must_use]
    pub fn phase_window_extent(&self) -> f64 {
        let mut max_coord = DEFAULT_PHASE_EXTENT;
        for val in [self.ic_y0.abs(), self.ic_v0.abs()] {
            if val.is_finite() {
                max_coord = max_coord.max(val * 1.35);
            }
        }
        for &[sx, sy] in &self.seeded_ic {
            if sx.is_finite() && sy.is_finite() {
                max_coord = max_coord.max(sx.abs() * 1.25).max(sy.abs() * 1.25);
            }
        }
        max_coord.clamp(DEFAULT_PHASE_EXTENT, 25.0)
    }

    /// Seeds a new $(x_0, y_0)$ orbit (maintaining a FIFO ring capped at [`MAX_SEEDED_ORBITS`]).
    pub fn seed_orbit(&mut self, ic: [f64; 2]) {
        if !ic[0].is_finite() || !ic[1].is_finite() {
            return;
        }
        if self.seeded_ic.len() >= MAX_SEEDED_ORBITS {
            self.seeded_ic.remove(0);
        }
        self.seeded_ic.push(ic);
        self.recalculate();
    }

    /// Clears all click-seeded phase-plane orbits.
    pub fn clear_seeded_orbits(&mut self) {
        self.seeded_ic.clear();
        self.seeded_trajectories.clear();
        self.bump_revision();
    }

    pub(super) fn recalculate_seeded_orbits(&mut self) {
        self.seeded_trajectories.clear();
        if !self.has_second_state() || self.seeded_ic.is_empty() {
            return;
        }
        let orbit_count = (1 + self.seeded_ic.len()).min(1 + MAX_SEEDED_ORBITS);
        let max_steps_each = (crate::presets::MAX_ODE_STEPS / orbit_count).max(1);
        let effective_dt = self
            .dt
            .max((self.total_time / max_steps_each as f64).max(1e-6));

        let custom_spec = self.build_custom_ode_spec();
        let px_expr = self.planar_x_input.cached_expr();
        let py_expr = self.planar_y_input.cached_expr();

        for &[sx, sy] in self.seeded_ic.iter().take(MAX_SEEDED_ORBITS) {
            let traj = simulate_ode(&OdeSimSpec {
                preset: self.preset,
                method: self.method,
                dt: effective_dt,
                total_time: self.total_time,
                param_k: self.param_k,
                param_r: self.param_r,
                param_cap_k: self.param_cap_k,
                ic_y0: sx,
                ic_v0: sy,
                matrix_a: self.matrix_a,
                custom_spec: custom_spec.as_ref(),
                planar_x_expr: px_expr,
                planar_y_expr: py_expr,
            });
            self.seeded_trajectories.push(traj);
        }
    }

    /// Evaluates the 2D quiver arrow field `(origins, tips)` capped strictly at $25 \times 25$ arrows.
    #[must_use]
    pub fn compute_quiver_arrows(&self) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        let n = self.quiver_density.clamp(2, MAX_QUIVER_GRID);
        let ext = self.phase_window_extent();
        let step = (2.0 * ext) / (n.saturating_sub(1).max(1) as f64);
        let p_map = build_ode_param_map(self.param_k, self.param_r, self.param_cap_k);
        let mut origins = Vec::with_capacity(n * n);
        let mut tips = Vec::with_capacity(n * n);

        for ix in 0..n {
            let x = -ext + ix as f64 * step;
            for iy in 0..n {
                let y = -ext + iy as f64 * step;
                let [dx, dy] = self.eval_planar_rhs_at(0.0, x, y, &p_map);
                let mag = dx.hypot(dy);
                if !dx.is_finite() || !dy.is_finite() || !mag.is_finite() || mag < 1e-12 {
                    continue;
                }
                let scale = 0.38 * step * (mag / (1.0 + mag)).max(0.28) / mag;
                let tip = [x + dx * scale, y + dy * scale];
                if tip[0].is_finite() && tip[1].is_finite() {
                    origins.push([x, y]);
                    tips.push(tip);
                }
            }
        }
        (origins, tips)
    }

    /// Computes marching-squares zero-contour line segments for $(x' = 0,\ y' = 0)$ nullclines.
    #[must_use]
    pub fn compute_nullcline_segments(&self) -> (Vec<NullclineSegment>, Vec<NullclineSegment>) {
        const GRID_N: usize = 36;
        let ext = self.phase_window_extent();
        let step = (2.0 * ext) / GRID_N as f64;
        let p_map = build_ode_param_map(self.param_k, self.param_r, self.param_cap_k);
        let mut grid = vec![[[0.0_f64; 2]; GRID_N + 1]; GRID_N + 1];
        for (ix, col) in grid.iter_mut().enumerate() {
            let x = -ext + ix as f64 * step;
            for (iy, cell) in col.iter_mut().enumerate() {
                *cell = self.eval_planar_rhs_at(0.0, x, -ext + iy as f64 * step, &p_map);
            }
        }
        let mut x_segs = Vec::new();
        let mut y_segs = Vec::new();
        for ix in 0..GRID_N {
            let x0 = -ext + ix as f64 * step;
            for iy in 0..GRID_N {
                let y0 = -ext + iy as f64 * step;
                let (c00, c10, c11, c01) = (
                    grid[ix][iy],
                    grid[ix + 1][iy],
                    grid[ix + 1][iy + 1],
                    grid[ix][iy + 1],
                );
                if let Some(s) =
                    march_cell_zero_segment(x0, y0, step, c00[0], c10[0], c11[0], c01[0])
                {
                    x_segs.push(s);
                }
                if let Some(s) =
                    march_cell_zero_segment(x0, y0, step, c00[1], c10[1], c11[1], c01[1])
                {
                    y_segs.push(s);
                }
            }
        }
        (x_segs, y_segs)
    }

    pub(super) fn render_phase_plane_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.add_space(8.0);
        ui.heading("Phase Space & Seeding");
        changed |= ui
            .checkbox(&mut self.phase_plane_mode, "2D Phase Portrait View")
            .changed();
        if !self.phase_plane_mode {
            return changed;
        }
        changed |= ui
            .checkbox(&mut self.show_quiver, "Show Quiver Vector Field")
            .changed();
        if self.show_quiver {
            let mut density = self.quiver_density.clamp(5, MAX_QUIVER_GRID);
            if ui
                .add(egui::Slider::new(&mut density, 5..=MAX_QUIVER_GRID).text("Quiver Grid"))
                .changed()
            {
                self.quiver_density = density.min(MAX_QUIVER_GRID);
                changed = true;
            }
        }
        changed |= ui
            .checkbox(&mut self.show_nullclines, "Show Nullclines (x'=0, y'=0)")
            .changed();
        ui.horizontal(|ui| {
            ui.label(format!(
                "Seeded Orbits: {}/{MAX_SEEDED_ORBITS}",
                self.seeded_ic.len()
            ));
            if !self.seeded_ic.is_empty() && ui.small_button("Clear Seeds").clicked() {
                self.clear_seeded_orbits();
                changed = true;
            }
        });
        changed
    }

    pub(super) fn render_phase_plane_canvas(&mut self, ui: &mut egui::Ui, changed: &mut bool) {
        ui.heading("2D Phase Portrait & Twin Timelines (Click Canvas to Seed Orbits)");
        if FiniteStateGuard::diverged(self) {
            let mut scratch_y = Vec::new();
            let mut scratch_v = Vec::new();
            recovery::render_recovery_ui(ui, self, changed, &mut scratch_y, &mut scratch_v);
        }

        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let phase_h = (ui.available_height().max(320.0) * 0.62).max(200.0);
        let clicked_seed = self.show_phase_plot(ui, &palette, phase_h);

        ui.add_space(4.0);
        let twin_h = ui.available_height().max(110.0);
        self.render_twin_timelines(ui, &palette, twin_h);

        if let Some(seed) = clicked_seed {
            self.seed_orbit(seed);
            *changed = true;
        }
    }

    fn show_phase_plot(
        &self,
        ui: &mut egui::Ui,
        palette: &crate::app::SemanticPalette,
        phase_h: f32,
    ) -> Option<[f64; 2]> {
        let (_, _, px_lbl, py_lbl) = self.active_state_labels();
        let (origins, tips) = if self.show_quiver {
            self.compute_quiver_arrows()
        } else {
            (Vec::new(), Vec::new())
        };
        let nullclines = self
            .show_nullclines
            .then(|| self.compute_nullcline_segments());
        let mut clicked_seed = None;

        Plot::new("ode_phase_plot")
            .height(phase_h)
            .data_aspect(1.0)
            .x_axis_label(px_lbl)
            .y_axis_label(py_lbl)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                if !origins.is_empty() {
                    let color = palette.series_color(4).gamma_multiply(0.55);
                    plot_ui.arrows(
                        Arrows::new("Vector Field", origins, tips)
                            .color(color)
                            .tip_length(4.5),
                    );
                }
                if let Some((ref x_segs, ref y_segs)) = nullclines {
                    render_nullcline_lines(plot_ui, x_segs, "x-nullcline (x'=0)", palette.warning);
                    render_nullcline_lines(plot_ui, y_segs, "y-nullcline (y'=0)", palette.success);
                }
                self.draw_phase_orbits(plot_ui, palette, px_lbl, py_lbl);
                if plot_ui.response().clicked() {
                    if let Some(pt) = plot_ui.pointer_coordinate() {
                        if pt.x.is_finite() && pt.y.is_finite() {
                            clicked_seed = Some([pt.x, pt.y]);
                        }
                    }
                }
            });
        clicked_seed
    }

    fn draw_phase_orbits(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &crate::app::SemanticPalette,
        px_lbl: &str,
        py_lbl: &str,
    ) {
        if self.comparison_mode {
            self.comparison_manager.render_on_plot(plot_ui);
        } else {
            let pts = finite_phase_pairs(&self.y_series, &self.v_series);
            if !pts.is_empty() {
                plot_ui.line(
                    Line::new(
                        format!("Primary ({px_lbl}, {py_lbl})"),
                        PlotPoints::new(pts),
                    )
                    .color(palette.series_color(0))
                    .width(2.0_f32),
                );
                if self.ic_y0.is_finite() && self.ic_v0.is_finite() {
                    plot_ui.points(
                        Points::new("Primary IC", vec![[self.ic_y0, self.ic_v0]])
                            .radius(4.5_f32)
                            .color(palette.series_color(0)),
                    );
                }
            }
        }
        for (idx, traj) in self.seeded_trajectories.iter().enumerate() {
            let pts = finite_phase_pairs(&traj.y_series, &traj.v_series);
            let Some(&ic) = pts.first() else { continue };
            let color = palette.series_color(idx + 1);
            let name = format!("Orbit {} ({:.2}, {:.2})", idx + 1, ic[0], ic[1]);
            plot_ui.line(
                Line::new(name, PlotPoints::new(pts))
                    .color(color)
                    .width(1.75_f32),
            );
            plot_ui.points(Points::new("", vec![ic]).radius(4.0_f32).color(color));
        }
    }

    fn render_twin_timelines(
        &self,
        ui: &mut egui::Ui,
        palette: &crate::app::SemanticPalette,
        height: f32,
    ) {
        let (t1_lbl, t2_lbl, _, _) = self.active_state_labels();
        ui.columns(2, |cols| {
            draw_timeline_plot(
                &mut cols[0],
                ("ode_twin_x_plot", t1_lbl, height, false),
                &self.display_y,
                &self.seeded_trajectories,
                palette,
            );
            draw_timeline_plot(
                &mut cols[1],
                ("ode_twin_y_plot", t2_lbl, height, true),
                &self.display_v,
                &self.seeded_trajectories,
                palette,
            );
        });
    }
}

fn finite_phase_pairs(xs: &[f64], ys: &[f64]) -> Vec<[f64; 2]> {
    xs.iter()
        .copied()
        .zip(ys.iter().copied())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| [x, y])
        .collect()
}

fn draw_timeline_plot(
    ui: &mut egui::Ui,
    meta: (&str, &str, f32, bool),
    primary: &std::sync::Arc<Vec<egui_plot::PlotPoint>>,
    seeded: &[SolverTrajectory],
    palette: &crate::app::SemanticPalette,
) {
    let (plot_id, label, height, second_component) = meta;
    let cap = crate::framework::MAX_DISPLAY_POLYLINE_POINTS;
    Plot::new(plot_id)
        .height(height)
        .x_axis_label("t")
        .y_axis_label(label)
        .show(ui, |plot_ui| {
            if !primary.is_empty() {
                plot_ui.line(
                    Line::new(
                        label,
                        PlotPoints::from_shared(std::sync::Arc::clone(primary)),
                    )
                    .color(palette.series_color(0)),
                );
            }
            for (idx, traj) in seeded.iter().enumerate() {
                let values = if second_component {
                    &traj.v_series
                } else {
                    &traj.y_series
                };
                let decimated =
                    crate::framework::decimate_xy_series_minmax(&traj.time_series, values, cap);
                plot_ui.line(
                    Line::new(
                        format!("Orbit {} {label}", idx + 1),
                        PlotPoints::from_shared(decimated),
                    )
                    .color(palette.series_color(idx + 1)),
                );
            }
        });
}

fn render_nullcline_lines(
    plot_ui: &mut egui_plot::PlotUi<'_>,
    segments: &[NullclineSegment],
    label: &str,
    color: egui::Color32,
) {
    for (idx, seg) in segments.iter().enumerate() {
        let name = if idx == 0 { label } else { "" };
        plot_ui.line(
            Line::new(name, PlotPoints::new(vec![seg[0], seg[1]]))
                .color(color)
                .style(LineStyle::Dashed { length: 5.0 })
                .width(1.4_f32),
        );
    }
}
