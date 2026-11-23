#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    AttractorMode, AttractorPlotter, LorenzRegimeRequest, LorenzRegimeSnapshot,
    SolverExecutionState, CORRELATION_EPSILON, REGIME_GRID_SIZE, REGIME_RHO_RANGE,
    REGIME_SIGMA_RANGE,
};
use crate::accessibility::AccessibleHoverText;
use crate::async_sim::{SimCommand, SimWorkerState};
use eframe::egui::{self, Color32, Stroke, Vec2};
use math_commons::math_kernel::types::flatten_2d_index;
use std::sync::atomic::Ordering;
use std::sync::Arc;

impl AttractorPlotter {
    /// Teleports `(rho, sigma)` onto a new parameter point and restarts the 3D trajectory cleanly.
    pub fn teleport_regime(&mut self, rho: f64, sigma: f64) {
        if !rho.is_finite() || !sigma.is_finite() {
            return;
        }
        let was_running = self.execution_state == SolverExecutionState::Running;
        self.mode = AttractorMode::Lorenz;
        self.system.rho = rho.clamp(REGIME_RHO_RANGE.0, REGIME_RHO_RANGE.1);
        self.system.sigma = sigma.clamp(REGIME_SIGMA_RANGE.0, REGIME_SIGMA_RANGE.1);
        self.reset();
        if was_running {
            let _ = self.try_resume();
        }
    }

    pub(super) fn pull_regime_atlas_snapshot(&mut self, ctx: &egui::Context) {
        let Some(snap) = self.regime_atlas.worker.update() else {
            return;
        };
        if let Ok(px) = snap.pixels.try_read() {
            if px.len() == snap.width * snap.height && snap.width > 0 {
                let img = egui::ColorImage::new([snap.width, snap.height], px.clone());
                let opts = egui::TextureOptions::NEAREST;
                if let Some(tex) = &mut self.regime_atlas.texture {
                    tex.set(img, opts);
                } else {
                    self.regime_atlas.texture =
                        Some(ctx.load_texture("lorenz_regime_atlas", img, opts));
                }
            }
        }
        if let Some(meta) = snap
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<LorenzRegimeSnapshot>())
        {
            self.regime_atlas.lambdas = Arc::clone(&meta.lambdas);
            self.regime_atlas.current_stride = meta.stride;
            if meta.is_final {
                self.regime_atlas.completed_generation = meta.generation;
            }
        }
    }

    pub(crate) fn sync_regime_atlas(&mut self, ctx: &egui::Context, high_contrast: bool) {
        self.pull_regime_atlas_snapshot(ctx);
        let safe_beta = if self.system.beta.is_finite() {
            self.system.beta.clamp(0.0, 10.0)
        } else {
            8.0 / 3.0
        };
        if safe_beta.to_bits() != self.regime_atlas.last_beta_bits
            || high_contrast != self.regime_atlas.high_contrast
        {
            self.regime_atlas.last_beta_bits = safe_beta.to_bits();
            self.regime_atlas.high_contrast = high_contrast;
            self.regime_atlas.request_generation =
                self.regime_atlas.request_generation.wrapping_add(1).max(1);
            *self
                .regime_atlas
                .shared_request
                .write()
                .unwrap_or_else(|e| e.into_inner()) = LorenzRegimeRequest {
                generation: self.regime_atlas.request_generation,
                beta: safe_beta,
                width: REGIME_GRID_SIZE.0,
                height: REGIME_GRID_SIZE.1,
                high_contrast,
            };
            self.regime_atlas
                .latest_generation
                .store(self.regime_atlas.request_generation, Ordering::Release);
            if self.regime_atlas.worker.is_running() {
                let _ = self
                    .regime_atlas
                    .worker
                    .send_command(SimCommand::Custom("cancel_regime".to_string()));
            }
        }
        if self.regime_atlas.completed_generation != self.regime_atlas.request_generation
            && self.regime_atlas.worker.state() == SimWorkerState::Ready
        {
            let _ = self.regime_atlas.worker.send_command(SimCommand::Start);
        }
        if self.regime_atlas.completed_generation != self.regime_atlas.request_generation
            || self.regime_atlas.worker.is_running()
        {
            ctx.request_repaint();
        }
    }

    pub(super) fn draw_custom_correlation_badge(&self, ui: &mut egui::Ui) {
        let corr_text = match self.latest_correlation_sum {
            Some(c) if !self.diverged() => format!("C(ε={CORRELATION_EPSILON:.1}): {c:.4}"),
            _ => format!("C(ε={CORRELATION_EPSILON:.1}): —"),
        };
        ui.small(corr_text);
    }

    fn draw_lyapunov_badges(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let (lyap_text, lyap_color) = if self.diverged() {
            ("λ_max: Halted (diverged)".to_string(), palette.error)
        } else if let Some(lam) = self.latest_lyapunov {
            let tag = if lam > 0.01 { "Chaotic" } else { "Stable" };
            let col = if lam > 0.01 {
                palette.series_color(1)
            } else {
                palette.series_color(2)
            };
            (format!("λ_max: {lam:+.3} ({tag})"), col)
        } else {
            ("λ_max: Pending…".to_string(), ui.visuals().text_color())
        };
        ui.colored_label(lyap_color, lyap_text);
        self.draw_custom_correlation_badge(ui);
        if self.regime_atlas.completed_generation != self.regime_atlas.request_generation {
            ui.small(format!(
                "Regime sweep: {}x pass",
                self.regime_atlas.current_stride
            ));
        }
    }

    pub(super) fn draw_lyapunov_badges_and_regime_atlas(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
    ) {
        self.draw_lyapunov_badges(ui);
        egui::CollapsingHeader::new("2D Regime Map (ρ × σ)")
            .default_open(true)
            .show(ui, |ui| {
                let desired = Vec2::new(ui.available_width().clamp(140.0, 210.0), 104.0);
                let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::click());
                if let Some(tex) = &self.regime_atlas.texture {
                    let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                    ui.painter().image(tex.id(), rect, uv, Color32::WHITE);
                } else {
                    ui.painter().rect_filled(rect, 2.0, Color32::from_gray(24));
                }
                let stroke = ui.visuals().widgets.noninteractive.bg_stroke;
                ui.painter()
                    .rect_stroke(rect, 2.0, stroke, egui::StrokeKind::Inside);

                let tx = ((self.system.rho - REGIME_RHO_RANGE.0)
                    / (REGIME_RHO_RANGE.1 - REGIME_RHO_RANGE.0))
                    .clamp(0.0, 1.0) as f32;
                let ty = (1.0
                    - (self.system.sigma - REGIME_SIGMA_RANGE.0)
                        / (REGIME_SIGMA_RANGE.1 - REGIME_SIGMA_RANGE.0))
                    .clamp(0.0, 1.0) as f32;
                let marker = egui::pos2(
                    rect.left() + tx * rect.width(),
                    rect.top() + ty * rect.height(),
                );
                ui.painter()
                    .circle_stroke(marker, 4.5_f32, Stroke::new(1.8_f32, Color32::WHITE));

                if let Some(pos) = resp.interact_pointer_pos().or_else(|| resp.hover_pos()) {
                    let rx = ((pos.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0) as f64;
                    let ry = (1.0 - (pos.y - rect.top()) / rect.height().max(1.0)).clamp(0.0, 1.0)
                        as f64;
                    let hover_rho =
                        REGIME_RHO_RANGE.0 + rx * (REGIME_RHO_RANGE.1 - REGIME_RHO_RANGE.0);
                    let hover_sigma =
                        REGIME_SIGMA_RANGE.0 + ry * (REGIME_SIGMA_RANGE.1 - REGIME_SIGMA_RANGE.0);
                    let gx = ((rx * (REGIME_GRID_SIZE.0 - 1) as f64).round() as usize)
                        .min(REGIME_GRID_SIZE.0 - 1);
                    let gy = (((1.0 - ry) * (REGIME_GRID_SIZE.1 - 1) as f64).round() as usize)
                        .min(REGIME_GRID_SIZE.1 - 1);
                    let cell_lam = self
                        .regime_atlas
                        .lambdas
                        .get(flatten_2d_index(gx, gy, REGIME_GRID_SIZE.0))
                        .copied()
                        .unwrap_or(0.0);
                    if resp.clicked() {
                        self.teleport_regime(hover_rho, hover_sigma);
                        ctx.request_repaint();
                    }
                    let tip = format!(
                        "ρ={hover_rho:.1}, σ={hover_sigma:.1}, λ≈{cell_lam:+.2} (click to teleport)"
                    );
                    let _ = resp.accessible_hover_text(tip);
                }
            });
    }
}
