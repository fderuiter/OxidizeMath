#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Poincaré trace-determinant $(\tau, \Delta)$ stability minimap and $2 \times 2$ linear
//! equilibrium classification for [`super::OdeSolverTool`].

use super::OdeSolverTool;
use crate::presets::OdePreset;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui::{self, Align2, Color32};
use egui_plot::{Line, LineStyle, Plot, PlotPoint, PlotPoints, Points, Text};

/// Tolerance around $\tau = 0$ (for $\Delta > 0$) classified as a neutral [`EquilibriumRegime::Center`].
pub const CENTER_TRACE_TOLERANCE: f64 = 1e-2;

/// Default $2 \times 2$ system matrix $A$ (damped harmonic spiral: $\tau = -0.4, \Delta = 1.0$).
pub const DEFAULT_MATRIX_A: [[f64; 2]; 2] = [[0.0, 1.0], [-1.0, -0.4]];

const MINIMAP_TAU_BOUND: f64 = 4.5;
const MINIMAP_DELTA_MIN: f64 = -3.0;
const MINIMAP_DELTA_MAX: f64 = 5.2;

/// Canonical topological equilibrium regimes on the Poincaré $(\tau, \Delta)$ plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EquilibriumRegime {
    /// $\Delta \le 0$: real eigenvalues of opposite sign (or degenerate zero determinant).
    Saddle,
    /// $\tau < 0,\ 0 < \Delta \le \tau^2 / 4$: two negative real eigenvalues.
    StableNode,
    /// $\tau > 0,\ 0 < \Delta \le \tau^2 / 4$: two positive real eigenvalues.
    UnstableNode,
    /// $\tau < 0,\ \Delta > \tau^2 / 4$: complex conjugate eigenvalues with $\text{Re}(\lambda) < 0$.
    StableSpiral,
    /// $\tau > 0,\ \Delta > \tau^2 / 4$: complex conjugate eigenvalues with $\text{Re}(\lambda) > 0$.
    UnstableSpiral,
    /// $|\tau| \le \epsilon,\ \Delta > 0$: purely imaginary conjugate eigenvalues.
    Center,
}

impl EquilibriumRegime {
    /// All six canonical Poincaré trace-determinant regimes.
    #[cfg(test)]
    pub const ALL: [Self; 6] = [
        Self::Saddle,
        Self::StableNode,
        Self::UnstableNode,
        Self::StableSpiral,
        Self::UnstableSpiral,
        Self::Center,
    ];

    /// Human-readable label for the equilibrium regime.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Saddle => "Saddle",
            Self::StableNode => "Stable Node",
            Self::UnstableNode => "Unstable Node",
            Self::StableSpiral => "Stable Spiral",
            Self::UnstableSpiral => "Unstable Spiral",
            Self::Center => "Center",
        }
    }
}

/// Computes trace $\tau = \text{tr}(A)$ and determinant $\Delta = \det(A)$ of a $2 \times 2$ matrix.
#[must_use]
pub fn trace_det_from_matrix(a: [[f64; 2]; 2]) -> (f64, f64) {
    let tau = a[0][0] + a[1][1];
    let delta = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    (tau, delta)
}

/// Builds the rational canonical companion matrix $A = [[0, 1], [-\Delta, \tau]]$ realizing $(\tau, \Delta)$.
#[must_use]
pub fn companion_matrix_from_trace_det(tau: f64, delta: f64) -> [[f64; 2]; 2] {
    [[0.0, 1.0], [-delta, tau]]
}

/// Classifies the equilibrium at trace $\tau$ and determinant $\Delta$.
#[must_use]
pub fn classify_equilibrium(tau: f64, delta: f64) -> EquilibriumRegime {
    if delta <= 0.0 {
        EquilibriumRegime::Saddle
    } else if tau.abs() <= CENTER_TRACE_TOLERANCE {
        EquilibriumRegime::Center
    } else if tau * tau - 4.0 * delta >= 0.0 {
        if tau < 0.0 {
            EquilibriumRegime::StableNode
        } else {
            EquilibriumRegime::UnstableNode
        }
    } else if tau < 0.0 {
        EquilibriumRegime::StableSpiral
    } else {
        EquilibriumRegime::UnstableSpiral
    }
}

/// Formats the eigenvalue pair $\lambda_{1,2} = \frac{\tau \pm \sqrt{\tau^2 - 4\Delta}}{2}$.
#[must_use]
pub fn format_eigenvalues(tau: f64, delta: f64) -> String {
    let disc = tau * tau - 4.0 * delta;
    if disc >= 0.0 {
        let root = disc.sqrt();
        let l1 = 0.5 * (tau + root);
        let l2 = 0.5 * (tau - root);
        format!("{l1:.3}, {l2:.3}")
    } else {
        let re = 0.5 * tau;
        let im = 0.5 * (-disc).sqrt();
        format!("{re:.3} ± {im:.3}i")
    }
}

fn regime_label_anchors() -> [(EquilibriumRegime, [f64; 2]); 6] {
    [
        (EquilibriumRegime::Saddle, [0.0, -1.6]),
        (EquilibriumRegime::StableNode, [-3.1, 0.85]),
        (EquilibriumRegime::UnstableNode, [3.1, 0.85]),
        (EquilibriumRegime::StableSpiral, [-2.0, 3.5]),
        (EquilibriumRegime::UnstableSpiral, [2.0, 3.5]),
        (EquilibriumRegime::Center, [0.0, 4.65]),
    ]
}

fn draw_poincare_boundaries_and_labels(
    plot_ui: &mut egui_plot::PlotUi<'_>,
    palette: &crate::app::SemanticPalette,
    text_color: Color32,
) {
    let parabola_pts: Vec<[f64; 2]> = (0..=64)
        .map(|i| {
            let tau = -MINIMAP_TAU_BOUND + (2.0 * MINIMAP_TAU_BOUND) * (i as f64 / 64.0);
            [tau, 0.25 * tau * tau]
        })
        .collect();
    plot_ui.line(
        Line::new(r"\Delta = \tau^2 / 4", PlotPoints::new(parabola_pts))
            .color(palette.series_color(2))
            .width(1.6_f32),
    );
    plot_ui.line(
        Line::new(
            r"\Delta = 0",
            PlotPoints::new(vec![[-MINIMAP_TAU_BOUND, 0.0], [MINIMAP_TAU_BOUND, 0.0]]),
        )
        .color(palette.series_color(1))
        .style(LineStyle::Dashed { length: 5.0 }),
    );
    plot_ui.line(
        Line::new(
            r"\tau = 0",
            PlotPoints::new(vec![[0.0, 0.0], [0.0, MINIMAP_DELTA_MAX]]),
        )
        .color(palette.series_color(3))
        .style(LineStyle::Dotted { spacing: 4.0 }),
    );

    let weak_label = text_color.gamma_multiply(0.85);
    for (regime, pos) in regime_label_anchors() {
        plot_ui.text(
            Text::new("", PlotPoint::new(pos[0], pos[1]), regime.label())
                .color(weak_label)
                .anchor(Align2::CENTER_CENTER),
        );
    }
}

impl OdeSolverTool {
    fn set_poincare_point(&mut self, tau: f64, delta: f64) -> bool {
        if !tau.is_finite() || !delta.is_finite() {
            return false;
        }
        let clamped_tau = tau.clamp(-10.0, 10.0);
        let clamped_delta = delta.clamp(-10.0, 25.0);
        self.preset = OdePreset::LinearSystem2D;
        self.matrix_a = companion_matrix_from_trace_det(clamped_tau, clamped_delta);
        self.phase_plane_mode = true;
        true
    }

    /// Applies a $(\tau, \Delta)$ coordinate from the Poincaré map, setting $A = [[0, 1], [-\Delta, \tau]]$,
    /// activating `LinearSystem2D` + phase-plane mode, and recalculating trajectories immediately.
    #[cfg(test)]
    pub fn apply_poincare_point(&mut self, tau: f64, delta: f64) {
        if self.set_poincare_point(tau, delta) {
            self.recalculate();
        }
    }

    /// Returns the active $(\tau, \Delta)$ pair for the current 2D linear or harmonic system.
    #[must_use]
    pub fn current_trace_det(&self) -> (f64, f64) {
        if self.preset == OdePreset::HarmonicOscillator {
            (0.0, self.param_k)
        } else {
            trace_det_from_matrix(self.matrix_a)
        }
    }

    pub(super) fn render_matrix_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label("Matrix A (x' = A·x):");
        egui::Grid::new("ode_matrix_a_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.matrix_a[0][0])
                            .prefix("a11: ")
                            .speed(0.05)
                            .range(-10.0..=10.0),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.matrix_a[0][1])
                            .prefix("a12: ")
                            .speed(0.05)
                            .range(-10.0..=10.0),
                    )
                    .changed();
                ui.end_row();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.matrix_a[1][0])
                            .prefix("a21: ")
                            .speed(0.05)
                            .range(-10.0..=10.0),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.matrix_a[1][1])
                            .prefix("a22: ")
                            .speed(0.05)
                            .range(-10.0..=10.0),
                    )
                    .changed();
                ui.end_row();
            });
        changed
    }

    pub(super) fn render_poincare_minimap(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.add_space(8.0);
        ui.heading("Poincaré (τ, Δ) Map");
        let (tau, delta) = self.current_trace_det();
        let regime = classify_equilibrium(tau, delta);
        let eig_str = format_eigenvalues(tau, delta);

        ui.group(|ui| {
            render_copyable_metric(ui, "Regime", regime.label());
            render_copyable_metric(ui, "τ / Δ", &format!("{tau:.2} / {delta:.2}"));
            render_copyable_metric(ui, "Eigenvalues", &eig_str);
        });

        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let text_color = ui.visuals().text_color();
        let mut picked: Option<[f64; 2]> = None;

        Plot::new("ode_poincare_minimap")
            .height(185.0)
            .allow_zoom(false)
            .allow_drag(false)
            .allow_scroll(false)
            .allow_boxed_zoom(false)
            .include_x(-MINIMAP_TAU_BOUND)
            .include_x(MINIMAP_TAU_BOUND)
            .include_y(MINIMAP_DELTA_MIN)
            .include_y(MINIMAP_DELTA_MAX)
            .x_axis_label(r"\tau")
            .y_axis_label(r"\Delta")
            .show(ui, |plot_ui| {
                draw_poincare_boundaries_and_labels(plot_ui, &palette, text_color);

                let marker_tau = tau.clamp(-MINIMAP_TAU_BOUND, MINIMAP_TAU_BOUND);
                let marker_delta = delta.clamp(MINIMAP_DELTA_MIN, MINIMAP_DELTA_MAX);
                plot_ui.points(
                    Points::new(r"(\tau, \Delta)", vec![[marker_tau, marker_delta]])
                        .radius(5.5_f32)
                        .color(palette.series_color(0)),
                );

                let resp = plot_ui.response();
                if resp.clicked() || resp.dragged() {
                    if let Some(pt) = plot_ui.pointer_coordinate() {
                        if pt.x.is_finite() && pt.y.is_finite() {
                            picked = Some([
                                pt.x.clamp(-MINIMAP_TAU_BOUND, MINIMAP_TAU_BOUND),
                                pt.y.clamp(MINIMAP_DELTA_MIN, MINIMAP_DELTA_MAX),
                            ]);
                        }
                    }
                }
            });

        if let Some([new_tau, new_delta]) = picked {
            changed |= self.set_poincare_point(new_tau, new_delta);
        }

        changed
    }
}
