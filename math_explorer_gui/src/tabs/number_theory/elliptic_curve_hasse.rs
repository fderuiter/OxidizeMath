//! Hasse-Weil Frobenius trace plot buffers, `egui_plot` renderer, control panel UI, and exports
//! for [`super::elliptic_curve_playground::EllipticCurvePlayground`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::{announce_live_status, AccessibleHoverText};
use crate::app::SemanticPalette;
use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport};
use crate::web_io::PlatformInstant;
use eframe::egui;
use egui_plot::export_primitives::ExportPrimitive;
use egui_plot::{Legend, Line, MarkerShape, Plot, PlotPoint, PlotPoints, Points};
use math_explorer::pure_math::elliptic_curves::{AffinePointFp, HasseSweepEntry};

use super::elliptic_curve_canvas::TorusCanvasOutcome;
use super::elliptic_curve_playground::{
    ActivePointSelector, EllipticCurvePlayground, MAX_PLAYGROUND_PRIME, MAX_SCALAR_K,
    MIN_PLAYGROUND_PRIME,
};

/// Preallocated plot series buffers for the Hasse-Weil Frobenius trace chart.
#[derive(Clone, Debug)]
pub struct HassePlotBuffers {
    /// Upper Hasse arch $+2\sqrt{q}$.
    pub upper_arch: Vec<PlotPoint>,
    /// Lower Hasse arch $-2\sqrt{q}$.
    pub lower_arch: Vec<PlotPoint>,
    /// Horizontal zero reference $a_q = 0$.
    pub zero_line: Vec<PlotPoint>,
    /// Non-singular Frobenius traces $(q, a_q)$.
    pub traces: Vec<PlotPoint>,
    /// Supersingular primes with $a_q = 0$.
    pub supersingular: Vec<PlotPoint>,
    /// Anomalous primes with $a_q = 1$ ($\#E(\mathbb{F}_q) = q$).
    pub anomalous: Vec<PlotPoint>,
    /// Highlighted marker for the currently active prime modulus $p$.
    pub active_prime: Vec<PlotPoint>,
}

impl HassePlotBuffers {
    /// Preallocates all plot series buffers to `capacity` to avoid hot-loop heap allocations.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            upper_arch: Vec::with_capacity(capacity),
            lower_arch: Vec::with_capacity(capacity),
            zero_line: Vec::with_capacity(2),
            traces: Vec::with_capacity(capacity),
            supersingular: Vec::with_capacity(capacity),
            anomalous: Vec::with_capacity(capacity),
            active_prime: Vec::with_capacity(1),
        }
    }

    /// Populates all Hasse-Weil plot buffers in-place from `entries` without reallocating.
    pub fn refresh(&mut self, entries: &[HasseSweepEntry], active_p: i64) {
        self.upper_arch.clear();
        self.lower_arch.clear();
        self.zero_line.clear();
        self.traces.clear();
        self.supersingular.clear();
        self.anomalous.clear();
        self.active_prime.clear();

        if let (Some(first), Some(last)) = (entries.first(), entries.last()) {
            self.zero_line.push(PlotPoint::new(first.p as f64, 0.0));
            self.zero_line.push(PlotPoint::new(last.p as f64, 0.0));
        }

        for entry in entries {
            let q = entry.p as f64;
            self.upper_arch.push(PlotPoint::new(q, entry.hasse_bound));
            self.lower_arch.push(PlotPoint::new(q, -entry.hasse_bound));
            if entry.is_singular {
                continue;
            }
            let pt = PlotPoint::new(q, entry.frobenius_trace as f64);
            self.traces.push(pt);
            if entry.frobenius_trace == 0 {
                self.supersingular.push(pt);
            }
            if entry.frobenius_trace == 1 {
                self.anomalous.push(pt);
            }
            if entry.p == active_p {
                self.active_prime.push(pt);
            }
        }
    }

    /// Builds tabular CSV/JSON datasets preserving Hasse arch exports even when active `p` is singular.
    #[must_use]
    pub fn build_datasets(
        &self,
        p: i64,
        is_singular: bool,
        points: &[AffinePointFp],
    ) -> Vec<(String, Vec<[f64; 2]>)> {
        let to_pairs = |pts: &[PlotPoint]| pts.iter().map(|pt| [pt.x, pt.y]).collect();
        let mut datasets = vec![
            ("Hasse +2*sqrt(q)".to_string(), to_pairs(&self.upper_arch)),
            ("Hasse -2*sqrt(q)".to_string(), to_pairs(&self.lower_arch)),
        ];
        if !self.traces.is_empty() {
            datasets.push(("Frobenius trace a_q".to_string(), to_pairs(&self.traces)));
        }
        if !is_singular {
            let pts: Vec<[f64; 2]> = points
                .iter()
                .filter_map(|pt| pt.coordinates().map(|(x, y)| [x as f64, y as f64]))
                .collect();
            if !pts.is_empty() {
                datasets.push((format!("Curve E(F_{p})"), pts));
            }
        }
        datasets
    }

    /// Builds headless SVG/PNG/GIF offscreen plot primitives from the Hasse-Weil trace buffers.
    #[must_use]
    pub fn build_primitives(&self) -> Vec<ExportPrimitive> {
        let to_pairs = |pts: &[PlotPoint]| pts.iter().map(|pt| [pt.x, pt.y]).collect();
        let palette = crate::app::palette::DARK_PALETTE;
        let mut prims = vec![
            ExportPrimitive::Polyline {
                name: "Hasse +2*sqrt(q)".to_string(),
                points: to_pairs(&self.upper_arch),
                color: palette.warning,
                stroke_width: 2.0_f32,
                closed: false,
                fill: None,
            },
            ExportPrimitive::Polyline {
                name: "Hasse -2*sqrt(q)".to_string(),
                points: to_pairs(&self.lower_arch),
                color: palette.warning,
                stroke_width: 2.0_f32,
                closed: false,
                fill: None,
            },
        ];
        if !self.traces.is_empty() {
            prims.push(ExportPrimitive::Points {
                name: "Frobenius trace a_q".to_string(),
                points: to_pairs(&self.traces),
                color: palette.series[0],
                radius: 3.5_f32,
            });
        }
        prims
    }
}

/// Renders the Hasse-Weil Frobenius trace arch chart $a_q = q + 1 - \#E(\mathbb{F}_q)$ inside `[-2*sqrt(q), 2*sqrt(q)]`.
pub fn render_hasse_plot(ui: &mut egui::Ui, buffers: &HassePlotBuffers, p_max: i64) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    ui.label(format!(
        "Hasse-Weil Frobenius Trace Arch  a_q = q + 1 - #E(F_q)  (5 <= q <= {p_max})"
    ));
    Plot::new("elliptic_curve_hasse_weil_plot")
        .legend(Legend::default())
        .x_axis_label("Prime modulus q")
        .y_axis_label("Frobenius trace a_q")
        .allow_scroll(false)
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new(
                    "Hasse +2*sqrt(q)",
                    PlotPoints::Borrowed(&buffers.upper_arch),
                )
                .color(palette.warning)
                .width(2.0_f32),
            );
            plot_ui.line(
                Line::new(
                    "Hasse -2*sqrt(q)",
                    PlotPoints::Borrowed(&buffers.lower_arch),
                )
                .color(palette.warning)
                .width(2.0_f32),
            );
            plot_ui.line(
                Line::new(
                    "Supersingular axis a_q = 0",
                    PlotPoints::Borrowed(&buffers.zero_line),
                )
                .color(palette.muted_text)
                .width(1.0_f32),
            );
            plot_ui.points(
                Points::new("Frobenius trace a_q", PlotPoints::Borrowed(&buffers.traces))
                    .color(palette.series[0])
                    .radius(3.4_f32)
                    .shape(MarkerShape::Circle),
            );
            plot_special_primes(plot_ui, buffers, &palette);
        });
}

fn plot_special_primes<'a>(
    plot_ui: &mut egui_plot::PlotUi<'a>,
    buffers: &'a HassePlotBuffers,
    palette: &SemanticPalette,
) {
    if !buffers.supersingular.is_empty() {
        plot_ui.points(
            Points::new(
                "Supersingular (a_q = 0)",
                PlotPoints::Borrowed(&buffers.supersingular),
            )
            .color(palette.series[4])
            .radius(5.2_f32)
            .shape(MarkerShape::Diamond),
        );
    }
    if !buffers.anomalous.is_empty() {
        plot_ui.points(
            Points::new(
                "Anomalous (a_q = 1)",
                PlotPoints::Borrowed(&buffers.anomalous),
            )
            .color(palette.error)
            .radius(5.0_f32)
            .shape(MarkerShape::Square),
        );
    }
    if !buffers.active_prime.is_empty() {
        plot_ui.points(
            Points::new(
                "Active prime p",
                PlotPoints::Borrowed(&buffers.active_prime),
            )
            .color(palette.accent)
            .radius(6.8_f32)
            .shape(MarkerShape::Circle),
        );
    }
}

pub(super) fn fmt_point(pt: AffinePointFp) -> String {
    match pt {
        AffinePointFp::Infinity => "O".to_string(),
        AffinePointFp::Finite { x, y } => format!("({x}, {y})"),
    }
}

impl EllipticCurvePlayground {
    pub(crate) fn select_curve_point_by_index(&mut self, ctx: &egui::Context, idx: usize) {
        if idx >= self.points_len {
            return;
        }
        match self.selector {
            ActivePointSelector::PointP => self.point_p = self.points_buf[idx],
            ActivePointSelector::PointQ => self.point_q = self.points_buf[idx],
        }
        self.ray_progress = 1.0_f32;
        self.recompute_all();
        announce_live_status(ctx, self.status_summary.clone());
    }

    pub(super) fn apply_canvas_outcome(
        &mut self,
        ctx: &egui::Context,
        outcome: TorusCanvasOutcome,
    ) {
        self.focused_canvas = outcome.focused_canvas_id;
        if outcome.toggle_selector {
            self.selector = match self.selector {
                ActivePointSelector::PointP => ActivePointSelector::PointQ,
                ActivePointSelector::PointQ => ActivePointSelector::PointP,
            };
            announce_live_status(ctx, format!("Active point selector: {:?}", self.selector));
        }
        if let Some(idx) = outcome.clicked_point_index {
            self.select_curve_point_by_index(ctx, idx);
        } else if outcome.arrow_delta != 0 && self.points_len > 0 {
            let cur = match self.selector {
                ActivePointSelector::PointP => self.point_p,
                ActivePointSelector::PointQ => self.point_q,
            };
            let pos = self.points_buf[..self.points_len]
                .iter()
                .position(|&pt| pt == cur)
                .unwrap_or(0);
            let next = (pos as isize + outcome.arrow_delta as isize)
                .rem_euclid(self.points_len as isize) as usize;
            self.select_curve_point_by_index(ctx, next);
        }
    }

    pub(super) fn render_controls_panel(&mut self, ui: &mut egui::Ui) {
        if let Some(action) =
            PlaybackTransport::in_thread("ec_fp_transport", self.name(), self.is_playing)
                .step_label("⏭ Step [k]P")
                .show(ui)
        {
            match action {
                PlaybackAction::Play => {
                    self.is_playing = true;
                    self.ray_progress = 0.1_f32;
                    self.last_frame_instant = Some(PlatformInstant::now());
                }
                PlaybackAction::Pause => self.is_playing = false,
                PlaybackAction::Step => {
                    self.step_orbit_hot_loop();
                    self.refresh_status_summary();
                    self.bump_revision();
                }
                PlaybackAction::Reset => *self = Self::default(),
            }
        }
        ui.separator();
        self.render_parameter_sliders(ui);
        ui.separator();
        self.render_torsion_sonification(ui);
        ui.separator();
        self.render_algebraic_readout(ui);
    }

    fn render_prime_stepper(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .button("◀ p")
                .accessible_hover_text("Previous prime modulus")
                .clicked()
            {
                self.step_prime(-1);
            }
            let mut p_edit = self.p;
            if ui
                .add(
                    egui::DragValue::new(&mut p_edit)
                        .range(MIN_PLAYGROUND_PRIME..=MAX_PLAYGROUND_PRIME),
                )
                .accessible_hover_text("Prime modulus p in 5..=997")
                .changed()
            {
                self.p = self.snap_nearest_prime(p_edit);
                self.recompute_all();
            }
            if ui
                .button("p ▶")
                .accessible_hover_text("Next prime modulus")
                .clicked()
            {
                self.step_prime(1);
            }
        });
    }

    fn render_parameter_sliders(&mut self, ui: &mut egui::Ui) {
        self.render_prime_stepper(ui);
        let mut changed = ui
            .add(egui::Slider::new(&mut self.a, -50..=50).text("a"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.b, -50..=50).text("b"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut self.hasse_p_max,
                    MIN_PLAYGROUND_PRIME..=MAX_PLAYGROUND_PRIME,
                )
                .text("Hasse p_max"),
            )
            .changed();
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.selector, ActivePointSelector::PointP, "Point P");
            ui.selectable_value(&mut self.selector, ActivePointSelector::PointQ, "Point Q");
            if !self.is_singular
                && ui
                    .button("Q = P (2P)")
                    .accessible_hover_text("Tangent doubling")
                    .clicked()
            {
                self.point_q = self.point_p;
                changed = true;
            }
        });
        let max_k = (self.p_order.max(1) as i64).clamp(1, MAX_SCALAR_K);
        changed |= ui
            .add(egui::Slider::new(&mut self.scalar_k, 1..=max_k).text("Scalar k"))
            .changed();
        if changed {
            self.ray_progress = 1.0_f32;
            self.recompute_all();
        }
    }

    fn render_algebraic_readout(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        if self.is_singular {
            ui.colored_label(
                palette.error,
                format!("Singular: 4a^3 + 27b^2 = 0 (mod {})", self.p),
            );
            return;
        }
        let sum = self.chord.map_or(AffinePointFp::Infinity, |c| c.sum);
        ui.label(format!(
            "#E(F_{}) = {}  |  a_p = {}  (|a_p| <= {:.2})",
            self.p,
            self.group_order,
            self.frobenius_trace,
            2.0 * (self.p as f64).sqrt()
        ));
        ui.label(format!(
            "P = {} (ord = {})  |  Q = {}",
            fmt_point(self.point_p),
            self.p_order,
            fmt_point(self.point_q)
        ));
        ui.label(format!(
            "P + Q = {}  |  [{}]P = {}",
            fmt_point(sum),
            self.scalar_k,
            fmt_point(self.scalar_kp)
        ));
    }
}

impl scientific_metadata::theory::TheoryDescribable for EllipticCurvePlayground {
    fn theory_description(&self) -> String {
        format!(
            "Short Weierstrass elliptic curve $E(\\mathbb{{F}}_{0}): y^2 \\equiv x^3 + {1}x + {2} \\pmod{{{0}}}$ with non-singular discriminant $\\Delta = -16(4a^3 + 27b^2) \\not\\equiv 0 \\pmod{{{0}}}$. Secant and tangent lines wrap geodesically across $(\\mathbb{{Z}}/{0}\\mathbb{{Z}})^2$, while Frobenius traces $a_q = q + 1 - \\#E(\\mathbb{{F}}_q)$ obey Hasse's bound $|a_q| \\le 2\\sqrt{{q}}$.",
            self.p, self.a, self.b
        )
    }
    fn theory_latex_equation(&self) -> Option<String> {
        Some("y^2 \\equiv x^3 + ax + b \\pmod{p}, \\quad |p + 1 - \\#E(\\mathbb{F}_p)| \\le 2\\sqrt{p}".into())
    }
    fn phonetic_description(&self) -> String {
        format!(
            "Elliptic curve y squared congruent to x cubed plus {} x plus {} modulo {}, with torus chord-and-tangent addition and Hasse-Weil arch.",
            self.a, self.b, self.p
        )
    }
    fn theory_citation(&self) -> String {
        "Silverman, The Arithmetic of Elliptic Curves (2009); Hasse, Zur Theorie der abstrakten elliptischen Funktionenkoerper (1936); Breuer, Divisibility of Coefficients of Modular Polynomials (2015)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Torus Secant & Hasse Arch over F_23",
    "domain": "number_theory",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "p": 23,
    "a": 1,
    "b": 1,
    "point_p": { "Finite": { "x": 0, "y": 1 } },
    "point_q": { "Finite": { "x": 6, "y": 4 } },
    "scalar_k": 5,
    "hasse_p_max": 251
  }
}"#;

inventory::submit! {
    crate::framework::StarterPresetMetadata {
        id: "elliptic_curve_fp_starter",
        title: "Elliptic Curve over F_23 & Hasse Arch",
        description: "Wrap-around chord-and-tangent group law on y^2 = x^3 + x + 1 (mod 23) paired with Frobenius trace Hasse bounds.",
        category: "Arithmetic Geometry",
        icon: "🍩",
        domain: "number_theory",
        tool_name: Some("EllipticCurvePlayground"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:modular_polynomials_review]
