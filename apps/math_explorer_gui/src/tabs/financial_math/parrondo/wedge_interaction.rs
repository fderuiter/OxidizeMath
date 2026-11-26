//! Interactive `(p_A, ε)` phase-wedge hover preview, keyboard cursor, and click teleportation
//! for [`super::ParrondoParadoxTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{
    solve_parrondo_ratchet_stack, WEDGE_EPS_MAX, WEDGE_EPS_MIN, WEDGE_PA_MAX, WEDGE_PA_MIN,
};
use super::{KellyEdgeBadge, ParrondoParadoxTool};
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility,
};
use crate::app::SemanticPalette;
use crate::async_sim::atlas::AtlasTeleportEvent;
use eframe::egui;
use eframe::egui::accesskit::Role;
use egui_plot::{Legend, Line, MarkerShape, Plot, PlotPoint, Points};
use math_explorer::pure_math::statistics::markov::parrondo::{
    scan_parrondo_phase_region, ParrondoParams, ParrondoPhaseGridConfig, ParrondoReport,
};
use std::hash::{Hash, Hasher};

const WEDGE_NUDGE_PA: f64 = 0.0025;
const WEDGE_NUDGE_EPS: f64 = 0.0025;

impl ParrondoParadoxTool {
    pub(super) fn rebuild_phase_wedge(&mut self) {
        let cfg = ParrondoPhaseGridConfig {
            modulus: self.modulus,
            p_b0_base: self.p_b0_base,
            p_b1_base: self.p_b1_base,
            p_a_min: WEDGE_PA_MIN,
            p_a_max: WEDGE_PA_MAX,
            epsilon_min: WEDGE_EPS_MIN,
            epsilon_max: WEDGE_EPS_MAX,
            p_a_steps: 29,
            epsilon_steps: 25,
        };
        self.paradox_scatter.clear();
        self.non_paradox_scatter.clear();
        if let Ok(pts) = scan_parrondo_phase_region(&cfg, &self.active_schedule) {
            for pt in &pts {
                let plot_pt = PlotPoint::new(pt.p_a, pt.epsilon);
                if pt.is_paradox {
                    self.paradox_scatter.push(plot_pt);
                } else {
                    self.non_paradox_scatter.push(plot_pt);
                }
            }
            super::plots::extract_phase_boundaries(
                &cfg,
                &pts,
                &mut self.boundary_game_a,
                &mut self.boundary_game_b,
                &mut self.boundary_schedule,
            );
            self.phase_points = pts;
        }
    }

    /// Teleports `(p_a_base, epsilon)` from effective `(p_A, ε)` wedge coordinates and recomputes the ratchet.
    pub fn teleport_phase_wedge(
        &mut self,
        p_a: f64,
        epsilon: f64,
        ctx: Option<&egui::Context>,
    ) -> bool {
        if !p_a.is_finite() || !epsilon.is_finite() {
            return false;
        }
        let eps = epsilon.clamp(WEDGE_EPS_MIN, WEDGE_EPS_MAX);
        let pa = p_a.clamp(WEDGE_PA_MIN, WEDGE_PA_MAX);
        let candidate_pa_base = (pa + eps).clamp(0.05, 0.95);
        if ParrondoParams::new(
            self.modulus,
            candidate_pa_base,
            self.p_b0_base,
            self.p_b1_base,
            eps,
        )
        .is_err()
        {
            return false;
        }
        self.epsilon = eps;
        self.p_a_base = candidate_pa_base;
        self.wedge_cursor = Some([self.p_a_base - self.epsilon, self.epsilon]);
        self.recompute_all();
        if let Some(c) = ctx {
            let gain = self.report.as_ref().map_or(0.0, |r| r.gain_schedule);
            let paradox = self.report.as_ref().is_some_and(ParrondoReport::is_paradox);
            announce_live_status(
                c,
                format!(
                    "Teleported Parrondo ratchet to p_A = {:.4}, epsilon = {:.4}; schedule drift {:+.5} ({}), paradox = {}",
                    self.p_a_base - self.epsilon,
                    self.epsilon,
                    gain,
                    self.kelly_schedule.badge_text(),
                    paradox
                ),
            );
            c.request_repaint();
        }
        true
    }

    pub(super) fn par_atlas_coords(&self, axis_keys: (&str, &str)) -> (f64, f64) {
        match axis_keys {
            ("p_a", "epsilon") => (self.p_a_base - self.epsilon, self.epsilon),
            ("p_a_base", "epsilon") => (self.p_a_base, self.epsilon),
            ("p_b0_base", "p_b1_base") => (self.p_b0_base, self.p_b1_base),
            _ => (self.p_a_base - self.epsilon, self.epsilon),
        }
    }

    pub(super) fn par_atlas_signature(&self, axis_keys: (&str, &str)) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.modulus.hash(&mut h);
        for &g in self.active_schedule.steps() {
            (g as u8).hash(&mut h);
        }
        match axis_keys {
            ("p_a", "epsilon") | ("p_a_base", "epsilon") => {
                self.p_b0_base.to_bits().hash(&mut h);
                self.p_b1_base.to_bits().hash(&mut h);
            }
            _ => {
                self.p_a_base.to_bits().hash(&mut h);
                self.epsilon.to_bits().hash(&mut h);
            }
        }
        h.finish()
    }

    pub(super) fn apply_parrondo_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        match (ev.x_key, ev.y_key) {
            ("p_a", "epsilon") => {
                let _ = self.teleport_phase_wedge(ev.x_val, ev.y_val, None);
            }
            ("p_a_base", "epsilon") => {
                self.epsilon = ev.y_val.clamp(-0.02, 0.045);
                let min_pa = (self.epsilon + 1e-3).max(0.35);
                let max_pa = (0.999 + self.epsilon).min(0.65);
                self.p_a_base = ev.x_val.clamp(min_pa, max_pa);
                let min_b = (self.epsilon + 1e-3).max(0.03);
                let max_b = (0.999 + self.epsilon).min(0.95);
                self.p_b0_base = self.p_b0_base.clamp(min_b, 0.50);
                self.p_b1_base = self.p_b1_base.clamp(0.50, max_b);
                self.wedge_cursor = Some([self.p_a_base - self.epsilon, self.epsilon]);
                self.recompute_all();
            }
            ("p_b0_base", "p_b1_base") => {
                let min_b = (self.epsilon + 1e-3).max(0.03);
                let max_b = (0.999 + self.epsilon).min(0.95);
                let b0 = ev.x_val.clamp(min_b, 0.50);
                let b1 = ev.y_val.clamp(0.50, max_b);
                if ParrondoParams::new(self.modulus, self.p_a_base, b0, b1, self.epsilon).is_ok() {
                    self.p_b0_base = b0;
                    self.p_b1_base = b1;
                    self.recompute_all();
                }
            }
            _ => return,
        }
        let target_x = self.num_steps as f64;
        let target_y = self
            .report
            .as_ref()
            .map_or(0.0, |r| r.gain_schedule * target_x);
        self.callouts.set_auto_callout(
            "atlas_teleport",
            [target_x, target_y],
            format!("Atlas: {}", ev.regime_label),
            [-110.0, -28.0],
        );
    }

    pub(super) fn render_phase_wedge_card(
        &mut self,
        ui: &mut egui::Ui,
        report: &ParrondoReport,
        palette: &SemanticPalette,
        height: f32,
    ) {
        let sched_str = self.active_schedule.as_string();
        ui.strong(format!(
            "(p_A, ε) Paradox Phase Wedge ({sched_str}) — Click to Teleport"
        ));
        let active_pa = self.p_a_base - self.epsilon;
        let active_pt = [PlotPoint::new(active_pa, self.epsilon)];
        let cursor_pt = self.wedge_cursor.map(|[cx, cy]| [PlotPoint::new(cx, cy)]);

        let mut hover_coord: Option<[f64; 2]> = None;
        let plot_resp = Plot::new("parrondo_phase_wedge_plot")
            .height((height - 22.0).max(110.0))
            .x_axis_label("Effective Game A Win Probability p_A = p_A,0 - ε")
            .y_axis_label("Bias Perturbation ε")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                self.draw_phase_wedge_series(plot_ui, palette, &active_pt, cursor_pt.as_ref());
                if let Some(pt) = plot_ui.pointer_coordinate() {
                    if (WEDGE_PA_MIN..=WEDGE_PA_MAX).contains(&pt.x)
                        && (WEDGE_EPS_MIN..=WEDGE_EPS_MAX).contains(&pt.y)
                    {
                        hover_coord = Some([pt.x, pt.y]);
                    }
                }
            });

        let clicked_coord = hover_coord.filter(|_| plot_resp.response.clicked());
        self.handle_wedge_keyboard(ui, &plot_resp.response, active_pa);
        if let Some([pa, eps]) = clicked_coord {
            let _ = self.teleport_phase_wedge(pa, eps, Some(ui.ctx()));
        }

        let inspect_target =
            hover_coord.or(self.wedge_cursor.filter(|_| plot_resp.response.has_focus()));
        self.render_wedge_preview_caption(
            ui,
            &plot_resp.response,
            palette,
            inspect_target,
            &sched_str,
        );

        let summary = format!(
            "Interactive phase wedge: active p_A = {:.4}, epsilon = {:.4}, paradox = {}, wedge samples = {}. Click or use Arrow keys and Enter to teleport.",
            self.p_a_base - self.epsilon,
            self.epsilon,
            report.is_paradox(),
            self.paradox_scatter.len()
        );
        register_canvas_accessibility(
            ui,
            &plot_resp.response,
            Role::Canvas,
            "Parrondo Paradox p_A versus epsilon Phase Wedge",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &plot_resp.response);
    }

    fn handle_wedge_keyboard(&mut self, ui: &egui::Ui, resp: &egui::Response, active_pa: f64) {
        lock_canvas_arrow_keys(ui, resp);
        if !resp.has_focus() {
            return;
        }
        let mut cur = self.wedge_cursor.unwrap_or([active_pa, self.epsilon]);
        self.wedge_cursor = Some(cur);
        let (dx, dy, activate) = ui.input(|i| {
            let dx = i32::from(i.key_pressed(egui::Key::ArrowRight))
                - i32::from(i.key_pressed(egui::Key::ArrowLeft));
            let dy = i32::from(i.key_pressed(egui::Key::ArrowUp))
                - i32::from(i.key_pressed(egui::Key::ArrowDown));
            let act = i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::Space);
            (dx, dy, act)
        });
        if dx != 0 || dy != 0 {
            cur[0] = (cur[0] + f64::from(dx) * WEDGE_NUDGE_PA).clamp(WEDGE_PA_MIN, WEDGE_PA_MAX);
            cur[1] = (cur[1] + f64::from(dy) * WEDGE_NUDGE_EPS).clamp(WEDGE_EPS_MIN, WEDGE_EPS_MAX);
            self.wedge_cursor = Some(cur);
            if let Some(msg) = self.format_wedge_preview(cur[0], cur[1]) {
                announce_live_status(ui.ctx(), msg);
            }
        }
        if activate {
            let _ = self.teleport_phase_wedge(cur[0], cur[1], Some(ui.ctx()));
        }
    }

    fn format_wedge_preview(&self, pa: f64, eps: f64) -> Option<String> {
        let sum = solve_parrondo_ratchet_stack(
            self.modulus,
            pa + eps,
            self.p_b0_base,
            self.p_b1_base,
            eps,
            self.active_schedule.steps(),
        )?;
        let ka = KellyEdgeBadge::evaluate(sum.gain_a).badge_text();
        let kb = KellyEdgeBadge::evaluate(sum.gain_b).badge_text();
        let ks = KellyEdgeBadge::evaluate(sum.gain_schedule).badge_text();
        let tag = if sum.is_paradox {
            "PARADOX"
        } else {
            "Non-Paradox"
        };
        Some(format!(
            "Preview (p_A={pa:.4}, ε={eps:.4}) [{tag}]: A {:+.4} ({ka}) | B {:+.4} ({kb}) | {} {:+.4} ({ks})",
            sum.gain_a,
            sum.gain_b,
            self.active_schedule.as_string(),
            sum.gain_schedule
        ))
    }

    fn render_wedge_preview_caption(
        &self,
        ui: &mut egui::Ui,
        resp: &egui::Response,
        palette: &SemanticPalette,
        inspect_target: Option<[f64; 2]>,
        sched_str: &str,
    ) {
        if let Some([pa, eps]) = inspect_target {
            if let Some(msg) = self.format_wedge_preview(pa, eps) {
                ui.colored_label(palette.accent, &msg);
                if resp.hovered() {
                    resp.clone().on_hover_text(msg);
                }
                return;
            }
        }
        ui.small(format!(
            "Click any (p_A, ε) point or focus plot & press Arrows + Enter to teleport {sched_str} parameters."
        ));
    }

    fn draw_phase_wedge_series<'a>(
        &'a self,
        plot_ui: &mut egui_plot::PlotUi<'a>,
        palette: &SemanticPalette,
        active_pt: &'a [PlotPoint; 1],
        cursor_pt: Option<&'a [PlotPoint; 1]>,
    ) {
        plot_ui.points(
            Points::new("Non-Paradox Grid", &self.non_paradox_scatter[..])
                .color(palette.series_color(5))
                .radius(2.2_f32),
        );
        plot_ui.points(
            Points::new(
                "Paradox Wedge (A<0, B<0, Sched>0)",
                &self.paradox_scatter[..],
            )
            .color(palette.success)
            .radius(4.0_f32),
        );
        for (name, pts, col, width) in [
            (
                "E[ΔW_A] = 0",
                &self.boundary_game_a[..],
                palette.series_color(0),
                1.6_f32,
            ),
            (
                "E[ΔW_B] = 0",
                &self.boundary_game_b[..],
                palette.series_color(1),
                1.6_f32,
            ),
            (
                "E[ΔW_sched] = 0",
                &self.boundary_schedule[..],
                palette.series_color(3),
                2.0_f32,
            ),
        ] {
            if !pts.is_empty() {
                plot_ui.line(Line::new(name, pts).color(col).width(width));
            }
        }
        if let Some(cpt) = cursor_pt {
            plot_ui.points(
                Points::new("Keyboard Cursor", &cpt[..])
                    .color(palette.warning)
                    .shape(MarkerShape::Cross)
                    .radius(6.0_f32),
            );
        }
        plot_ui.points(
            Points::new("Active (p_A, ε)", &active_pt[..])
                .color(palette.accent)
                .shape(MarkerShape::Diamond)
                .radius(6.5_f32),
        );
    }
}
