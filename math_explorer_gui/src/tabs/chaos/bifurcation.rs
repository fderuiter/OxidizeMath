#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::{AccessibleHoverText, AccessibleTheoryHover};
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::InteractiveTool;
use crate::reflective_ui::AnimatedSlider;
use eframe::egui;
use egui_plot::{HLine, Line, Plot, PlotPoint, PlotPoints, Points, VLine};
use math_explorer::physics::chaos::metrics::logistic_lyapunov;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

#[path = "bifurcation/runner.rs"]
mod runner;
pub use runner::*;

const fn default_selected_r() -> f64 { 3.83 }
const fn default_orbit_x0() -> f64 { 0.1 }
const fn default_show_lyapunov() -> bool { true }

/// Interactive Logistic Map bifurcation diagram coupled to live `logistic_lyapunov`,
/// cobweb, and $x_n$ orbit inspectors via progressive `BackgroundWorker` sweeps.
pub struct BifurcationDiagram {
    pub(crate) r_min: f64,
    pub(crate) r_max: f64,
    pub(crate) steps: usize,
    pub(crate) selected_r: f64,
    pub(crate) orbit_x0: f64,
    pub(crate) show_lyapunov: bool,
    pub(crate) points: Arc<Vec<PlotPoint>>,
    pub(crate) lyapunov_points: Arc<Vec<PlotPoint>>,
    pub(crate) raw_lyapunov: Arc<Vec<[f64; 2]>>,
    pub(crate) sweep_progress: Option<(usize, usize)>,
    pub(crate) dirty: bool,
    pub(crate) request_generation: u64,
    pub(crate) completed_generation: u64,
    pub(crate) shared_request: Arc<RwLock<BifurcationSweepRequest>>,
    pub(crate) latest_generation: Arc<AtomicU64>,
    pub(crate) worker: BackgroundWorker<BifurcationRunner>,
}

impl Default for BifurcationDiagram {
    fn default() -> Self {
        let r_min = 2.5;
        let r_max = 4.0;
        let steps = 500;
        let (pts, lyap, raw) = compute_bifurcation_and_lyapunov_sync(r_min, r_max, steps);
        let initial_req = BifurcationSweepRequest {
            generation: 0,
            r_min,
            r_max,
            steps,
        };
        let shared_request = Arc::new(RwLock::new(initial_req));
        let latest_generation = Arc::new(AtomicU64::new(0));
        let worker = BackgroundWorker::new(BifurcationRunner::new(
            Arc::clone(&shared_request),
            Arc::clone(&latest_generation),
            pts.clone(),
            lyap.clone(),
            raw.clone(),
        ));
        Self {
            r_min,
            r_max,
            steps,
            selected_r: default_selected_r(),
            orbit_x0: default_orbit_x0(),
            show_lyapunov: true,
            points: Arc::new(pts),
            lyapunov_points: Arc::new(lyap),
            raw_lyapunov: Arc::new(raw),
            sweep_progress: None,
            dirty: false,
            request_generation: 0,
            completed_generation: 0,
            shared_request,
            latest_generation,
            worker,
        }
    }
}

impl BifurcationDiagram {
    /// Synchronously recomputes the bifurcation attractor cloud and `logistic_lyapunov` curve.
    pub fn recompute(&mut self) {
        self.sanitize_bounds();
        let (pts, lyap, raw) =
            compute_bifurcation_and_lyapunov_sync(self.r_min, self.r_max, self.steps);
        self.points = Arc::new(pts);
        self.lyapunov_points = Arc::new(lyap);
        self.raw_lyapunov = Arc::new(raw);
        self.sweep_progress = None;
        self.dirty = false;
        self.completed_generation = self.request_generation;
    }

    fn sanitize_bounds(&mut self) {
        if !self.r_min.is_finite() {
            self.r_min = 2.5;
        }
        if !self.r_max.is_finite() {
            self.r_max = 4.0;
        }
        self.r_min = self.r_min.clamp(0.0, 3.99);
        self.r_max = self.r_max.clamp((self.r_min + 0.01).min(4.0), 4.0);
        if self.r_min >= self.r_max {
            self.r_min = (self.r_max - 0.01).max(0.0);
        }
        self.steps = self.steps.clamp(100, 2000);
        if !self.selected_r.is_finite() {
            self.selected_r = default_selected_r();
        }
        self.selected_r = self.selected_r.clamp(self.r_min, self.r_max);
        if !self.orbit_x0.is_finite() {
            self.orbit_x0 = default_orbit_x0();
        }
        self.orbit_x0 = self.orbit_x0.clamp(0.01, 0.99);
    }

    /// Selects a growth-rate parameter $r$ on the bifurcation axis and clamps it to valid bounds.
    pub fn select_r(&mut self, r: f64) {
        if r.is_finite() {
            self.selected_r = r.clamp(self.r_min, self.r_max).clamp(0.0, 4.0);
        }
    }

    /// Evaluates the live Lyapunov exponent $\lambda(r)$ at `selected_r`.
    #[must_use]
    pub fn selected_lyapunov(&self) -> f64 {
        logistic_lyapunov(
            self.selected_r.clamp(0.0, 4.0),
            BIFURCATION_LYAPUNOV_X0,
            BIFURCATION_LYAPUNOV_ITERS,
        )
    }

    fn sync_worker_snapshot(&mut self) {
        let Some(snapshot) = self.worker.update() else {
            return;
        };
        if let Some(data) = snapshot
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<BifurcationSweepSnapshot>())
        {
            if !data.points.is_empty() || data.is_final {
                self.points = Arc::clone(&data.points);
                self.lyapunov_points = Arc::clone(&data.lyapunov_points);
                self.raw_lyapunov = Arc::clone(&data.raw_lyapunov);
            }
            if data.is_final {
                self.completed_generation = data.generation;
                self.sweep_progress = None;
            } else {
                self.sweep_progress = Some((data.completed_steps, data.total_steps + 1));
            }
        }
    }

    fn dispatch_sweep_request(&mut self, ctx: &egui::Context) {
        if self.dirty {
            self.dirty = false;
            self.sanitize_bounds();
            self.request_generation = self.request_generation.wrapping_add(1).max(1);
            let req = BifurcationSweepRequest {
                generation: self.request_generation,
                r_min: self.r_min,
                r_max: self.r_max,
                steps: self.steps,
            };
            *self
                .shared_request
                .write()
                .unwrap_or_else(|e| e.into_inner()) = req;
            self.latest_generation
                .store(self.request_generation, Ordering::Release);

            match self.worker.state() {
                SimWorkerState::Ready => {
                    let _ = self.worker.send_command(SimCommand::Start);
                }
                SimWorkerState::Running => {
                    let _ = self
                        .worker
                        .send_command(SimCommand::Custom("cancel_sweep".to_string()));
                }
                SimWorkerState::Faulted | SimWorkerState::Terminated => {
                    let shared = Arc::clone(&self.shared_request);
                    let latest = Arc::clone(&self.latest_generation);
                    self.worker = BackgroundWorker::new(BifurcationRunner::new(
                        shared,
                        latest,
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                    ));
                    let _ = self.worker.send_command(SimCommand::Start);
                }
                SimWorkerState::Uninitialized | SimWorkerState::Stepping => {}
            }
        }
        if self.completed_generation != self.request_generation && self.worker.state().is_ready() {
            let _ = self.worker.send_command(SimCommand::Start);
        }
        if self.completed_generation != self.request_generation || self.worker.is_running() {
            ctx.request_repaint();
        }
    }

    fn draw_lyapunov_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.checkbox(&mut self.show_lyapunov, "Overlay Lyapunov λ(r)");
        let (sel_lo, sel_hi) = (self.r_min, self.r_max);
        self.selected_r = self.selected_r.clamp(sel_lo, sel_hi);
        ui.add(egui::Slider::new(&mut self.selected_r, sel_lo..=sel_hi).text("Selected r"));
        ui.add(egui::Slider::new(&mut self.orbit_x0, 0.01..=0.99).text("Orbit Seed (x₀)"));

        let lambda = self.selected_lyapunov();
        let regime = if lambda > 0.0 {
            "Chaotic (λ > 0)"
        } else {
            "Stable / Periodic (λ ≤ 0)"
        };
        ui.label(format!("λ({:.3}) = {:+.4} — {regime}", self.selected_r, lambda));
        if let Some((done, total)) = self.sweep_progress {
            ui.small(format!("Sweeping r: {done}/{total}"));
        }
        ui.separator();
        ui.small("Click bifurcation plot to inspect cobweb & x_n orbit.");
    }

    fn draw_side_panel(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "bifurcation_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading("Logistic Map");
            ui.separator();

            let mut sweep_changed = ui
                .add(
                    AnimatedSlider::new(&mut self.r_min, 0.0..=3.99)
                        .text("Min r")
                        .id_salt("bifurcation_r_min"),
                )
                .changed();

            let min_bound = (self.r_min + 0.01).min(4.0);
            if self.r_max < min_bound {
                self.r_max = min_bound;
            }
            sweep_changed |= ui
                .add(
                    AnimatedSlider::new(&mut self.r_max, min_bound..=4.0)
                        .text("Max r")
                        .id_salt("bifurcation_r_max"),
                )
                .changed();
            sweep_changed |= ui
                .add(egui::Slider::new(&mut self.steps, 100..=2000).text("Resolution (steps)"))
                .changed();

            if ui
                .button("▶ Recompute")
                .accessible_hover_text(
                    "Regenerate the bifurcation diagram with the current parameters",
                )
                .clicked()
                || sweep_changed
            {
                self.sanitize_bounds();
                self.dirty = true;
            }
            self.draw_lyapunov_controls(ui);
        });
    }

    fn draw_inspector_plots(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        let bottom_h = ui.available_height().max(100.0);
        let series = build_cobweb_and_orbit(self.selected_r, self.orbit_x0, COBWEB_ORBIT_STEPS);
        ui.columns(2, |cols| {
            Plot::new("logistic_cobweb_plot")
                .height(bottom_h)
                .x_axis_label("x_n")
                .y_axis_label("x_{n+1}")
                .show(&mut cols[0], |plot_ui| {
                    plot_ui.line(
                        Line::new("f_r(x) = r·x·(1-x)", PlotPoints::new(series.parabola))
                            .color(palette.series_color(0)),
                    );
                    plot_ui.line(
                        Line::new("y = x", PlotPoints::new(vec![[0.0, 0.0], [1.0, 1.0]]))
                            .color(palette.series_color(2)),
                    );
                    plot_ui.line(
                        Line::new("Cobweb", PlotPoints::new(series.cobweb))
                            .color(palette.series_color(1)),
                    );
                });
            Plot::new("logistic_orbit_plot")
                .height(bottom_h)
                .x_axis_label("Iteration (n)")
                .y_axis_label("State (x_n)")
                .show(&mut cols[1], |plot_ui| {
                    plot_ui.line(
                        Line::new("Orbit x_n", PlotPoints::new(series.orbit))
                            .color(palette.series_color(3)),
                    );
                });
        });
    }

    fn draw_central_panel(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let top_h = (ui.available_height().max(240.0) * 0.56).max(130.0);
            let mut clicked_r: Option<f64> = None;

            let plot_resp = Plot::new("bifurcation_plot")
                .height(top_h)
                .x_axis_label("Growth Rate (r)")
                .y_axis_label("Population (x) / Lyapunov (λ)")
                .show(ui, |plot_ui| {
                    plot_ui.points(
                        Points::new("Attractor", &self.points[..])
                            .radius(1.0_f32)
                            .color(palette.series_color(0)),
                    );
                    if self.show_lyapunov {
                        plot_ui.hline(
                            HLine::new("Stability Threshold (λ = 0)", 0.0)
                                .color(palette.series_color(2)),
                        );
                        plot_ui.line(
                            Line::new("Lyapunov λ(r)", &self.lyapunov_points[..])
                                .color(palette.series_color(1)),
                        );
                    }
                    plot_ui.vline(
                        VLine::new("Selected r", self.selected_r).color(palette.series_color(3)),
                    );
                    if plot_ui.response().clicked() && !plot_ui.response().dragged() {
                        clicked_r = plot_ui.pointer_coordinate().map(|c| c.x);
                    }
                });
            plot_resp.response.accessible_theory_hover(self);
            if let Some(r_val) = clicked_r {
                self.select_r(r_val);
            }
            ui.separator();
            self.draw_inspector_plots(ui, &palette);
        });
    }
}

fn is_default_selected_r(v: &f64) -> bool {
    (*v - default_selected_r()).abs() < 1e-12
}

fn is_default_orbit_x0(v: &f64) -> bool {
    (*v - default_orbit_x0()).abs() < 1e-12
}

const fn is_default_show_lyapunov(v: &bool) -> bool {
    *v
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BifurcationState {
    r_min: f64,
    r_max: f64,
    steps: usize,
    #[serde(
        default = "default_selected_r",
        skip_serializing_if = "is_default_selected_r"
    )]
    selected_r: f64,
    #[serde(
        default = "default_orbit_x0",
        skip_serializing_if = "is_default_orbit_x0"
    )]
    orbit_x0: f64,
    #[serde(
        default = "default_show_lyapunov",
        skip_serializing_if = "is_default_show_lyapunov"
    )]
    show_lyapunov: bool,
}

impl InteractiveTool for BifurcationDiagram {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Bifurcation Diagram"
    }

    fn domain(&self) -> &'static str {
        "chaos"
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let attractor: Vec<[f64; 2]> = self.points.iter().map(|p| [p.x, p.y]).collect();
        let lyapunov: Vec<[f64; 2]> = self.raw_lyapunov.as_ref().clone();
        vec![
            ("Attractor".to_string(), attractor),
            ("Lyapunov Exponent".to_string(), lyapunov),
        ]
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&BifurcationState {
            r_min: self.r_min,
            r_max: self.r_max,
            steps: self.steps,
            selected_r: self.selected_r,
            orbit_x0: self.orbit_x0,
            show_lyapunov: self.show_lyapunov,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: BifurcationState =
            serde_json::from_str(state).map_err(|e| format!("Invalid bifurcation state: {e}"))?;
        if !s.r_min.is_finite()
            || !s.r_max.is_finite()
            || !(0.0..=4.0).contains(&s.r_min)
            || !(0.0..=4.0).contains(&s.r_max)
            || s.r_min >= s.r_max
            || !(100..=2000).contains(&s.steps)
            || !s.selected_r.is_finite()
            || !(0.0..=4.0).contains(&s.selected_r)
            || !s.orbit_x0.is_finite()
            || !(0.0..=1.0).contains(&s.orbit_x0)
        {
            return Err("Bifurcation r bounds or step count invalid".into());
        }
        self.r_min = s.r_min;
        self.r_max = s.r_max;
        self.steps = s.steps;
        self.selected_r = s.selected_r.clamp(s.r_min, s.r_max);
        self.orbit_x0 = s.orbit_x0.clamp(0.01, 0.99);
        self.show_lyapunov = s.show_lyapunov;
        self.recompute();
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.sync_worker_snapshot();
        self.draw_side_panel(ui);
        self.draw_central_panel(ui);
        self.dispatch_sweep_request(&ctx);
    }
}

// [cite:algorithmic_information_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BifurcationDiagram",
        display_name: "Bifurcation Diagram",
        domain: "chaos",
        tags: &[],
        build: || Box::new(BifurcationDiagram::default()),
    }
}

const STARTER_PRESET_JSON: &str = r#"{"metadata":{"schema_version":"1.0","title":"Bifurcation Diagram","domain":"chaos","timestamp":"2026-11-16T00:00:00Z"},"config":{"r_min":2.5,"r_max":4.0,"steps":800}}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "bifurcation",
        title: "Bifurcation Diagram",
        description: "x_{n+1} = r·x_n·(1 - x_n): Period-doubling cascade towards deterministic chaos.",
        category: "Non-Linear Dynamics",
        icon: "🌿",
        domain: "chaos",
        tool_name: Some("BifurcationDiagram"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

#[cfg(test)]
#[path = "bifurcation/tests.rs"]
mod tests;
