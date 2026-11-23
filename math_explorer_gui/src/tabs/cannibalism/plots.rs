#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{CannibalismRegime, CannibalismToolState};
use crate::app::SemanticPalette;
use eframe::egui::{self, Color32, ColorImage, TextureOptions};
use egui_plot::{HLine, Legend, Line, Plot, PlotImage, PlotPoint, Points, VLine};
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::applied::cannibalism::{
    AgeStructuredConfig, AgeStructuredSolver, PredationRegime, TriboliumLpaModel, MAX_GRID_SIZE,
};

/// Number of historical time columns retained in the scrolling 2D Lexis `(t, a)` carpet.
pub const LEXIS_HISTORY_COLS: usize = 96;

/// Preallocated rolling 2D Lexis `(t, a)` density buffer and time-series workspace.
pub struct LexisCarpetBuffer {
    /// Flattened `(col, age_idx)` scalar log-density buffer of capacity `LEXIS_HISTORY_COLS * MAX_GRID_SIZE`.
    pub cells: Vec<f64>,
    /// Active age-bin height matching `solver.config().grid_size`.
    pub age_rows: usize,
    /// Number of populated history columns (`<= LEXIS_HISTORY_COLS`).
    pub filled_cols: usize,
    /// Ring-buffer write cursor in `0..LEXIS_HISTORY_COLS`.
    pub write_col: usize,
    /// Simulation time recorded for each column in the ring buffer.
    pub col_times: [f64; LEXIS_HISTORY_COLS],
    /// Adult population recorded for each column in the ring buffer.
    pub adult_history: [f64; LEXIS_HISTORY_COLS],
    /// Juvenile population recorded for each column in the ring buffer.
    pub juvenile_history: [f64; LEXIS_HISTORY_COLS],
    /// Reusable pixel staging buffer for `ColorImage` uploads.
    pixel_scratch: Vec<Color32>,
}

impl Default for LexisCarpetBuffer {
    fn default() -> Self {
        Self {
            cells: vec![0.0; LEXIS_HISTORY_COLS * MAX_GRID_SIZE],
            age_rows: 100,
            filled_cols: 0,
            write_col: 0,
            col_times: [0.0; LEXIS_HISTORY_COLS],
            adult_history: [0.0; LEXIS_HISTORY_COLS],
            juvenile_history: [0.0; LEXIS_HISTORY_COLS],
            pixel_scratch: Vec::with_capacity(LEXIS_HISTORY_COLS * 256),
        }
    }
}

impl LexisCarpetBuffer {
    /// Clears history for a given `age_rows` grid resolution without reallocating.
    pub fn reset(&mut self, age_rows: usize) {
        self.age_rows = age_rows.clamp(16, MAX_GRID_SIZE);
        self.filled_cols = 0;
        self.write_col = 0;
        self.cells.fill(0.0);
        self.col_times.fill(0.0);
        self.adult_history.fill(0.0);
        self.juvenile_history.fill(0.0);
    }

    /// Pushes the solver's current age-density slice into the next column of the Lexis ring buffer.
    pub fn push_snapshot(&mut self, solver: &AgeStructuredSolver) {
        let rows = solver.config().grid_size.min(MAX_GRID_SIZE);
        self.age_rows = rows;
        let col = self.write_col;
        let density = solver.density();
        for (a_idx, &val) in density.iter().take(rows).enumerate() {
            let idx = flatten_2d_index(col, a_idx, LEXIS_HISTORY_COLS);
            self.cells[idx] = (1.0 + val.max(0.0)).ln();
        }
        self.col_times[col] = solver.time();
        self.adult_history[col] = solver.adult_population();
        self.juvenile_history[col] = solver.juvenile_population();
        self.write_col = (col + 1) % LEXIS_HISTORY_COLS;
        self.filled_cols = (self.filled_cols + 1).min(LEXIS_HISTORY_COLS);
    }

    /// Chronologically ordered `(t, A(t))` and `(t, J(t))` trajectories from the ring buffer.
    pub fn ordered_trajectories(&self) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        let count = self.filled_cols;
        let mut adults = Vec::with_capacity(count);
        let mut juveniles = Vec::with_capacity(count);
        let start = if count < LEXIS_HISTORY_COLS {
            0
        } else {
            self.write_col
        };
        for offset in 0..count {
            let col = (start + offset) % LEXIS_HISTORY_COLS;
            let t = self.col_times[col];
            adults.push([t, self.adult_history[col]]);
            juveniles.push([t, self.juvenile_history[col]]);
        }
        (adults, juveniles)
    }

    /// Builds a `ColorImage` of the scrolling `(t, a)` Lexis plane using `flatten_2d_index`.
    pub fn build_color_image(&mut self, high_contrast: bool) -> ColorImage {
        let width = self.filled_cols.max(1);
        let height = self.age_rows.max(1);
        self.pixel_scratch.clear();
        self.pixel_scratch.resize(width * height, Color32::BLACK);

        let start = if self.filled_cols < LEXIS_HISTORY_COLS {
            0
        } else {
            self.write_col
        };
        let mut peak = 1.0e-6_f64;
        for x in 0..self.filled_cols {
            let ring_col = (start + x) % LEXIS_HISTORY_COLS;
            for a_idx in 0..height {
                let v = self.cells[flatten_2d_index(ring_col, a_idx, LEXIS_HISTORY_COLS)];
                if v > peak {
                    peak = v;
                }
            }
        }

        for x in 0..self.filled_cols {
            let ring_col = (start + x) % LEXIS_HISTORY_COLS;
            for a_idx in 0..height {
                let v = self.cells[flatten_2d_index(ring_col, a_idx, LEXIS_HISTORY_COLS)];
                let norm = (v / peak).clamp(0.0, 1.0);
                let rgb = if high_contrast {
                    heatmap_color_high_contrast(norm)
                } else {
                    heatmap_color(norm)
                };
                let img_y = height - 1 - a_idx;
                self.pixel_scratch[flatten_2d_index(x, img_y, width)] =
                    Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
            }
        }
        ColorImage::new([width, height], self.pixel_scratch.clone())
    }
}

/// Evaluates one parameter slice of the bifurcation diagram without mutating the live solver.
pub fn compute_bifurcation_slice(
    state: &CannibalismToolState,
    slice_idx: usize,
    out_points: &mut Vec<[f64; 2]>,
) {
    let total = state.sweep_points.max(2);
    let frac = slice_idx as f64 / (total - 1) as f64;
    match state.regime {
        CannibalismRegime::DiekmannHopf => {
            let beta = 8.0 + frac * 62.0;
            let cfg = AgeStructuredConfig {
                grid_size: 48,
                max_age: 5.0,
                maturation_age: 2.0,
                natural_mortality: state.mu0,
                beta0: beta,
                regime: PredationRegime::Diekmann {
                    vulnerable_age: 0.75,
                    attack_rate: state.c_cannibalism,
                    handling_time: 0.03,
                },
            };
            let mut solver = AgeStructuredSolver::new(cfg);
            solver.step_n(420);
            let mut min_a = f64::INFINITY;
            let mut max_a = 0.0_f64;
            for _ in 0..160 {
                solver.step();
                let a = solver.adult_population();
                min_a = min_a.min(a);
                max_a = max_a.max(a);
            }
            out_points.push([beta, min_a]);
            out_points.push([beta, max_a]);
        }
        CannibalismRegime::TriboliumChaos => {
            let c_pa = frac * 1.15;
            let lpa = TriboliumLpaModel::two_cycle_preset(c_pa);
            let mut orbit = [1.5, 0.5, 1.0];
            for _ in 0..320 {
                orbit = lpa.step(orbit);
            }
            for _ in 0..32 {
                orbit = lpa.step(orbit);
                if orbit[2].is_finite() {
                    out_points.push([c_pa, orbit[2]]);
                }
            }
        }
    }
}

/// Renders the scrolling 2D Lexis `(t, a)` heatmap inside an `egui_plot::Plot`.
pub fn render_lexis_plot(
    ui: &mut egui::Ui,
    lexis: &mut LexisCarpetBuffer,
    texture: &mut Option<egui::TextureHandle>,
    max_age: f64,
    maturation_age: f64,
    high_contrast: bool,
) {
    if texture.is_none() {
        let image = lexis.build_color_image(high_contrast);
        *texture = Some(ui.ctx().load_texture(
            "cannibalism_lexis_carpet",
            image,
            TextureOptions::LINEAR,
        ));
    }
    let palette = SemanticPalette::from_visuals(ui.visuals());
    let (adults, _) = lexis.ordered_trajectories();
    let (t_min, t_max) = match (adults.first(), adults.last()) {
        (Some(first), Some(last)) if last[0] > first[0] => (first[0], last[0]),
        _ => (0.0, 10.0),
    };
    let span_t = (t_max - t_min).max(1.0e-3);
    let center_t = 0.5 * (t_min + t_max);
    let center_a = 0.5 * max_age;

    Plot::new("cannibalism_lexis_plot")
        .height(235.0)
        .show_grid(false)
        .x_axis_label("Time t")
        .y_axis_label("Cohort Age a")
        .legend(Legend::default())
        .show(ui, |plot_ui| {
            if let Some(tex) = texture.as_ref() {
                plot_ui.image(PlotImage::new(
                    "Log Cohort Density ln(1 + n(t, a))",
                    tex.id(),
                    PlotPoint::new(center_t, center_a),
                    [span_t as f32, max_age as f32],
                ));
            }
            plot_ui.hline(
                HLine::new("Maturation Threshold", maturation_age)
                    .color(palette.warning)
                    .width(1.5_f32),
            );
        });
}

fn render_age_pyramid_column(
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
    solver: &AgeStructuredSolver,
) {
    let profile: Vec<[f64; 2]> = solver
        .ages()
        .iter()
        .zip(solver.density().iter())
        .map(|(&a, &n)| [a, n])
        .collect();
    ui.strong("Age-Pyramid Density n(t, a)");
    Plot::new("cannibalism_pyramid_and_traj")
        .height(220.0)
        .legend(Legend::default())
        .x_axis_label("Cohort Age a")
        .y_axis_label("Density n(t, a)")
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("Age Profile n(t, a)", profile)
                    .color(palette.series_color(0))
                    .width(2.0_f32),
            );
            plot_ui.vline(
                VLine::new("Maturation Age", solver.config().maturation_age).color(palette.warning),
            );
            if let Some(eq) = solver.diekmann_equilibrium() {
                plot_ui.hline(
                    HLine::new(format!("Diekmann Equilibrium A* = {eq:.3}"), eq)
                        .color(palette.success)
                        .width(1.5_f32),
                );
            }
        });
}

fn render_bifurcation_column(
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
    state: &CannibalismToolState,
    bifurcation_points: &[[f64; 2]],
) {
    let (bif_title, x_lbl, marker_x) = match state.regime {
        CannibalismRegime::DiekmannHopf => (
            "Diekmann Hopf Bifurcation (min/max A vs β₀)",
            "Fecundity β₀",
            state.beta0,
        ),
        CannibalismRegime::TriboliumChaos => (
            "Tribolium LPA Orbit Bifurcation (A vs c_pa)",
            "Callow Cannibalism Rate",
            state.c_pa,
        ),
    };
    ui.strong(bif_title);
    Plot::new("cannibalism_bifurcation_plot")
        .height(220.0)
        .legend(Legend::default())
        .x_axis_label(x_lbl)
        .y_axis_label("Asymptotic Adult Population A")
        .show(ui, |plot_ui| {
            plot_ui.points(
                Points::new("Orbit Branches", bifurcation_points.to_vec())
                    .color(palette.series_color(1))
                    .radius(2.2_f32),
            );
            plot_ui.vline(
                VLine::new("Active Parameter", marker_x)
                    .color(palette.error)
                    .width(1.5_f32),
            );
        });
}

/// Renders instantaneous age-pyramid `n(t, a)` and adult trajectory / bifurcation plots.
pub fn render_secondary_plots(
    ui: &mut egui::Ui,
    state: &CannibalismToolState,
    solver: &AgeStructuredSolver,
    lexis: &LexisCarpetBuffer,
    bifurcation_points: &[[f64; 2]],
) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    let (adults, juveniles) = lexis.ordered_trajectories();
    ui.columns(2, |cols| {
        render_age_pyramid_column(&mut cols[0], &palette, solver);
        render_bifurcation_column(&mut cols[1], &palette, state, bifurcation_points);
    });

    Plot::new("cannibalism_time_series_plot")
        .height(145.0)
        .legend(Legend::default())
        .x_axis_label("Time t")
        .y_axis_label("Stage Abundance")
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("Adults A(t)", adults)
                    .color(palette.series_color(2))
                    .width(2.0_f32),
            );
            plot_ui.line(
                Line::new("Juveniles J(t)", juveniles)
                    .color(palette.series_color(3))
                    .width(1.6_f32),
            );
            if let Some(eq) = solver.diekmann_equilibrium() {
                plot_ui.hline(HLine::new("Equilibrium A*", eq).color(palette.success));
            }
        });
}
