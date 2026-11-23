//! Interactive finite-field short Weierstrass elliptic curve playground ($E(\mathbb{F}_p)$)
//! with wrap-around torus chord/tangent rays and Hasse-Weil Frobenius trace sweeps.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::fmt::Write as _;
use std::hash::{Hash, Hasher};

use eframe::egui;
use egui_plot::export_primitives::ExportPrimitive;
use math_explorer::pure_math::elliptic_curves::{
    sweep_hasse_bounds, AffinePointFp, ChordStepFp, HasseSweepEntry, WeierstrassCurveFp,
};
use math_explorer::pure_math::number_theory::primes::{is_prime, primes_up_to};
use scientific_metadata::theory::TheoryDescribable;

use super::elliptic_curve_canvas::{
    populate_torus_ray_segments, render_torus_canvas, TorusCanvasModel, TorusCanvasOutcome,
};
use super::elliptic_curve_hasse::{fmt_point, render_hasse_plot, HassePlotBuffers};
use crate::accessibility::AccessibleHoverText;
use crate::framework::{
    scoped_panel_id, ExportCapabilities, InteractiveTool, PaneSidePanel, ToolMetadata,
    ToolStateBounds,
};
use crate::web_io::PlatformInstant;

/// Minimum prime modulus supported in the interactive torus playground.
pub const MIN_PLAYGROUND_PRIME: i64 = 5;
/// Maximum prime modulus supported in the interactive torus playground.
pub const MAX_PLAYGROUND_PRIME: i64 = 997;
pub(super) const MAX_SCALAR_K: i64 = 2048;

const MAX_CURVE_POINTS_CAP: usize = 1064;
const MAX_HASSE_PRIMES_CAP: usize = 170;
const MAX_RAY_SEGMENTS_CAP: usize = 64;

/// Which generator/operand point pointer clicks and arrow keys update on the torus canvas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivePointSelector {
    /// Select or step primary point $P$.
    PointP,
    /// Select or step secondary point $Q$.
    PointQ,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct EllipticCurvePlaygroundState {
    p: i64,
    a: i64,
    b: i64,
    point_p: AffinePointFp,
    point_q: AffinePointFp,
    scalar_k: i64,
    hasse_p_max: i64,
}

/// Interactive finite-field elliptic curve group law and Hasse-Weil Frobenius trace explorer.
pub struct EllipticCurvePlayground {
    pub(crate) p: i64,
    pub(crate) a: i64,
    pub(crate) b: i64,
    pub(crate) point_p: AffinePointFp,
    pub(crate) point_q: AffinePointFp,
    pub(crate) scalar_k: i64,
    pub(crate) hasse_p_max: i64,
    pub(crate) selector: ActivePointSelector,
    pub(crate) is_playing: bool,
    pub(crate) is_singular: bool,
    pub(crate) group_order: usize,
    pub(crate) frobenius_trace: i64,
    pub(crate) p_order: usize,
    pub(crate) scalar_kp: AffinePointFp,
    pub(crate) chord: Option<ChordStepFp>,
    pub(super) ray_progress: f32,
    orbit_step_accum: f32,
    pub(super) last_frame_instant: Option<PlatformInstant>,
    primes_table: Vec<i64>,
    pub(crate) points_buf: Vec<AffinePointFp>,
    pub(crate) points_len: usize,
    pub(crate) orbit_buf: Vec<AffinePointFp>,
    pub(crate) orbit_len: usize,
    pub(crate) hasse_buf: Vec<HasseSweepEntry>,
    pub(crate) hasse_len: usize,
    pub(crate) ray_segments: Vec<([f32; 2], [f32; 2])>,
    pub(crate) plot_buffers: HassePlotBuffers,
    pub(crate) status_summary: String,
    pub(super) focused_canvas: Option<egui::Id>,
    revision: u64,
}

impl Default for EllipticCurvePlayground {
    fn default() -> Self {
        let primes_table: Vec<i64> = primes_up_to(MAX_PLAYGROUND_PRIME as u64)
            .into_iter()
            .filter(|&q| q >= MIN_PLAYGROUND_PRIME as u64)
            .map(|q| q as i64)
            .collect();
        let dummy = HasseSweepEntry {
            p: 5,
            is_singular: false,
            group_order: 0,
            frobenius_trace: 0,
            hasse_bound: 0.0,
        };
        let mut tool = Self {
            p: 17,
            a: 2,
            b: 2,
            point_p: AffinePointFp::Finite { x: 5, y: 1 },
            point_q: AffinePointFp::Finite { x: 6, y: 3 },
            scalar_k: 3,
            hasse_p_max: 199,
            selector: ActivePointSelector::PointP,
            is_playing: false,
            is_singular: false,
            group_order: 0,
            frobenius_trace: 0,
            p_order: 1,
            scalar_kp: AffinePointFp::Infinity,
            chord: None,
            ray_progress: 1.0_f32,
            orbit_step_accum: 0.0_f32,
            last_frame_instant: None,
            primes_table,
            points_buf: vec![AffinePointFp::Infinity; MAX_CURVE_POINTS_CAP],
            points_len: 0,
            orbit_buf: vec![AffinePointFp::Infinity; MAX_CURVE_POINTS_CAP],
            orbit_len: 0,
            hasse_buf: vec![dummy; MAX_HASSE_PRIMES_CAP],
            hasse_len: 0,
            ray_segments: Vec::with_capacity(MAX_RAY_SEGMENTS_CAP),
            plot_buffers: HassePlotBuffers::with_capacity(MAX_HASSE_PRIMES_CAP),
            status_summary: String::with_capacity(256),
            focused_canvas: None,
            revision: 1,
        };
        tool.recompute_all();
        tool
    }
}

impl EllipticCurvePlayground {
    pub(super) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }

    pub(crate) fn recompute_all(&mut self) {
        self.recompute_curve_hot_loop();
        self.refresh_status_summary();
        self.bump_revision();
    }

    /// Recomputes curve enumeration, chord/tangent rays, torsion orbit, and Hasse sweep
    /// using preallocated slices with zero dynamic heap allocations.
    pub(crate) fn recompute_curve_hot_loop(&mut self) {
        self.p = self.snap_nearest_prime(self.p);
        self.hasse_p_max = self
            .hasse_p_max
            .clamp(MIN_PLAYGROUND_PRIME, MAX_PLAYGROUND_PRIME)
            .max(self.p);

        if let Ok(curve) = WeierstrassCurveFp::new(self.p, self.a, self.b) {
            self.populate_nonsingular_curve(curve);
        } else {
            self.is_singular = true;
            self.points_len = 0;
            self.orbit_len = 0;
            self.group_order = 0;
            self.frobenius_trace = 0;
            self.p_order = 1;
            self.scalar_kp = AffinePointFp::Infinity;
            self.chord = None;
            self.ray_segments.clear();
        }

        let count = self
            .primes_table
            .partition_point(|&q| q <= self.hasse_p_max);
        self.hasse_len = sweep_hasse_bounds(
            self.a,
            self.b,
            &self.primes_table[..count],
            &mut self.hasse_buf,
        )
        .unwrap_or(0);
        self.plot_buffers
            .refresh(&self.hasse_buf[..self.hasse_len], self.p);
    }

    fn populate_nonsingular_curve(&mut self, curve: WeierstrassCurveFp) {
        self.is_singular = false;
        self.points_len = curve.enumerate_points(&mut self.points_buf).unwrap_or(0);
        if !curve.contains_point(self.point_p) {
            self.point_p = self.points_buf.get(1).copied().unwrap_or_default();
        }
        if !curve.contains_point(self.point_q) {
            let idx = if self.points_len > 2 {
                2
            } else {
                1.min(self.points_len.saturating_sub(1))
            };
            self.point_q = self.points_buf[idx];
        }
        self.group_order = self.points_len;
        self.frobenius_trace = self.p + 1 - (self.group_order as i64);
        self.p_order = curve.point_order(self.point_p).unwrap_or(1);
        self.scalar_k = self.scalar_k.clamp(1, MAX_SCALAR_K);
        self.scalar_kp = curve
            .scalar_mul(self.scalar_k, self.point_p)
            .unwrap_or_default();
        self.orbit_len = curve
            .torsion_orbit(self.point_p, &mut self.orbit_buf)
            .unwrap_or(0);
        self.chord = curve.chord_step(self.point_p, self.point_q).ok();
        self.rebuild_ray_segments();
    }

    fn rebuild_ray_segments(&mut self) {
        self.ray_segments.clear();
        if let (Some(step), Some(start)) = (self.chord, self.point_p.coordinates()) {
            populate_torus_ray_segments(
                self.p,
                start,
                &step,
                self.ray_progress,
                MAX_RAY_SEGMENTS_CAP,
                &mut self.ray_segments,
            );
        }
    }

    /// Steps the cyclic subgroup multiplier `k` by one step using preallocated buffers (zero heap allocations).
    pub(crate) fn step_orbit_hot_loop(&mut self) {
        let Ok(curve) = WeierstrassCurveFp::new(self.p, self.a, self.b) else {
            return;
        };
        self.scalar_k = (self.scalar_k % (self.p_order.max(1) as i64)) + 1;
        self.scalar_kp = curve
            .scalar_mul(self.scalar_k, self.point_p)
            .unwrap_or_default();
        self.ray_progress = 1.0_f32;
        self.rebuild_ray_segments();
    }

    pub(super) fn snap_nearest_prime(&self, candidate: i64) -> i64 {
        let c = candidate.clamp(MIN_PLAYGROUND_PRIME, MAX_PLAYGROUND_PRIME);
        let pos = self.primes_table.partition_point(|&q| q < c);
        let hi = self.primes_table.get(pos).copied();
        let lo = pos
            .checked_sub(1)
            .and_then(|i| self.primes_table.get(i).copied());
        match (lo, hi) {
            (Some(l), Some(h)) if (c - l).abs() <= (h - c).abs() => l,
            (_, Some(h)) => h,
            (Some(l), None) => l,
            (None, None) => 17,
        }
    }

    pub(super) fn step_prime(&mut self, delta: isize) {
        let pos = self.primes_table.partition_point(|&q| q < self.p);
        let max_idx = self.primes_table.len().saturating_sub(1) as isize;
        let next = ((pos as isize) + delta).clamp(0, max_idx) as usize;
        if let Some(&next_p) = self.primes_table.get(next) {
            self.p = next_p;
            self.ray_progress = 1.0_f32;
            self.recompute_all();
        }
    }

    pub(crate) fn refresh_status_summary(&mut self) {
        self.status_summary.clear();
        if self.is_singular {
            let _ = write!(
                &mut self.status_summary,
                "Singular curve y^2 = x^3 + {}x + {} over F_{}: discriminant 4a^3 + 27b^2 = 0 mod {}",
                self.a, self.b, self.p, self.p
            );
            return;
        }
        let sum = self.chord.map_or(AffinePointFp::Infinity, |c| c.sum);
        let _ = write!(
            &mut self.status_summary,
            "E(F_{}): #E = {}, a_p = {}, P = {}, Q = {}, P+Q = {}, [{}]P = {} (ord = {})",
            self.p,
            self.group_order,
            self.frobenius_trace,
            fmt_point(self.point_p),
            fmt_point(self.point_q),
            fmt_point(sum),
            self.scalar_k,
            fmt_point(self.scalar_kp),
            self.p_order
        );
    }

    fn advance_animation(&mut self, ctx: &egui::Context) {
        let now = PlatformInstant::now();
        let dt = self.last_frame_instant.map_or(0.016_f32, |prev| {
            prev.elapsed().as_secs_f32().clamp(0.0_f32, 0.1_f32)
        });
        self.last_frame_instant = Some(now);
        if !self.is_playing || self.is_singular {
            return;
        }
        if self.ray_progress < 1.0_f32 {
            self.ray_progress = (self.ray_progress + dt * 1.6_f32).min(1.0_f32);
            self.rebuild_ray_segments();
        } else {
            self.orbit_step_accum += dt;
            if self.orbit_step_accum >= 0.45_f32 {
                self.orbit_step_accum = 0.0_f32;
                self.step_orbit_hot_loop();
                self.ray_progress = 0.25_f32;
                self.rebuild_ray_segments();
                self.refresh_status_summary();
                self.bump_revision();
            }
        }
        ctx.request_repaint();
    }
}

impl InteractiveTool for EllipticCurvePlayground {
    fn name(&self) -> &'static str {
        "Elliptic Curve Playground"
    }
    fn domain(&self) -> &'static str {
        "number_theory"
    }
    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }
    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::new(4096, 8, 16)
    }
    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (
            self.p,
            self.a,
            self.b,
            self.point_p,
            self.point_q,
            self.scalar_k,
            self.hasse_p_max,
            self.revision,
        )
            .hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&EllipticCurvePlaygroundState {
            p: self.p,
            a: self.a,
            b: self.b,
            point_p: self.point_p,
            point_q: self.point_q,
            scalar_k: self.scalar_k,
            hasse_p_max: self.hasse_p_max,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let s: EllipticCurvePlaygroundState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid elliptic curve state JSON: {e}"))?;
        if !(MIN_PLAYGROUND_PRIME..=MAX_PLAYGROUND_PRIME).contains(&s.p) || !is_prime(s.p as u64) {
            return Err(format!("Modulus p = {} must be a prime in 5..=997", s.p));
        }
        if !(-MAX_PLAYGROUND_PRIME..=MAX_PLAYGROUND_PRIME).contains(&s.a)
            || !(-MAX_PLAYGROUND_PRIME..=MAX_PLAYGROUND_PRIME).contains(&s.b)
            || !(1..=MAX_SCALAR_K).contains(&s.scalar_k)
            || !(MIN_PLAYGROUND_PRIME..=MAX_PLAYGROUND_PRIME).contains(&s.hasse_p_max)
        {
            return Err("Curve parameters out of allowed bounds".into());
        }
        let curve = WeierstrassCurveFp::new(s.p, s.a, s.b).map_err(|e| e.to_string())?;
        if !curve.contains_point(s.point_p) || !curve.contains_point(s.point_q) {
            return Err("Selected point P or Q does not lie on E(F_p)".into());
        }
        self.p = s.p;
        self.a = s.a;
        self.b = s.b;
        self.point_p = s.point_p;
        self.point_q = s.point_q;
        self.scalar_k = s.scalar_k;
        self.hasse_p_max = s.hasse_p_max.max(s.p);
        self.is_playing = false;
        self.ray_progress = 1.0_f32;
        self.recompute_all();
        Ok(())
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.plot_buffers.build_datasets(
            self.p,
            self.is_singular,
            &self.points_buf[..self.points_len],
        )
    }

    fn export_plot_primitives(&self) -> Vec<ExportPrimitive> {
        self.plot_buffers.build_primitives()
    }

    fn step_simulation(&mut self) {
        self.step_orbit_hot_loop();
        self.refresh_status_summary();
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.advance_animation(ui.ctx());
        PaneSidePanel::left(scoped_panel_id(ui, "ec_fp_controls")).show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading(self.name());
                let help_btn = ui
                    .add(egui::Button::new("❓").small())
                    .accessible_hover_text("Toggle Theory Context Portal");
                if help_btn.clicked() {
                    let key = egui::Id::new("SHOW_THEORY_PORTAL");
                    let next = !ui.ctx().data(|d| d.get_temp::<bool>(key).unwrap_or(false));
                    crate::framework::record_theory_portal_toggle(ui.ctx(), help_btn.id, next);
                }
            });
            ui.separator();
            self.render_controls_panel(ui);
        });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let model = TorusCanvasModel {
                p: self.p,
                a: self.a,
                b: self.b,
                is_singular: self.is_singular,
                points: &self.points_buf[..self.points_len],
                orbit: &self.orbit_buf[..self.orbit_len],
                point_p: self.point_p,
                point_q: self.point_q,
                scalar_kp: self.scalar_kp,
                scalar_k: self.scalar_k,
                chord: self.chord,
                ray_segments: &self.ray_segments,
                status_summary: &self.status_summary,
            };
            let mut outcome = TorusCanvasOutcome::default();
            if ui.available_width() >= 680.0_f32 {
                ui.columns(2, |cols| {
                    outcome = render_torus_canvas(&mut cols[0], &model, self);
                    render_hasse_plot(&mut cols[1], &self.plot_buffers, self.hasse_p_max);
                });
            } else {
                let half_h = (ui.available_height() * 0.52_f32).max(220.0_f32);
                ui.allocate_ui(egui::Vec2::new(ui.available_width(), half_h), |top| {
                    outcome = render_torus_canvas(top, &model, self);
                });
                ui.separator();
                render_hasse_plot(ui, &self.plot_buffers, self.hasse_p_max);
            }
            self.apply_canvas_outcome(ui.ctx(), outcome);
        });
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

inventory::submit! {
    ToolMetadata {
        id: "EllipticCurvePlayground",
        display_name: "Elliptic Curve Playground",
        domain: "number_theory",
        tags: &["elliptic curve", "finite field", "torus", "hasse bound", "frobenius trace"],
        build: || Box::new(EllipticCurvePlayground::default()),
    }
}

// [cite:modular_polynomials_review]
