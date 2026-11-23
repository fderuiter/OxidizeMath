#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:graph_parameters_rust]

use eframe::egui::{self, Pos2, Rect, Stroke};
use math_explorer::pure_math::analysis::pde::wave_2d::SpectralMode2D;

use super::ir_export::{
    PickupDragTarget, IR_DURATION_MAX_SECS, IR_DURATION_MIN_SECS, PICKUP_COORD_MAX,
    PICKUP_COORD_MIN,
};
use super::{ChladniWaveTankTool, GRID_SIZE};
use crate::app::SemanticPalette;
use crate::audio::mappings::wave_plate::{chladni_wave_tank_frame, MAX_CHLADNI_MODES};
use crate::audio::wav::{WavSampleFormat, WavSampleRate};
use crate::audio::{render_sonification_panel, SonificationFrame};

pub(crate) const SONIFICATION_TOOL_ID: &str = "chladni_wave_tank";
pub(crate) const MAX_CUSTOM_CLAMPS: usize = 16;
pub(crate) const CLAMP_COORD_MIN: f64 = 0.02;
pub(crate) const CLAMP_COORD_MAX: f64 = 0.98;

#[derive(Debug, Clone)]
pub(crate) struct ChladniSonificationState {
    pub(crate) top_modes: [SpectralMode2D; MAX_CHLADNI_MODES],
    pub(crate) mode_count: usize,
    pub(crate) pending_impulse: f32,
    pub(crate) latest_frame: SonificationFrame,
}

impl Default for ChladniSonificationState {
    fn default() -> Self {
        Self {
            top_modes: [SpectralMode2D::default(); MAX_CHLADNI_MODES],
            mode_count: 0,
            pending_impulse: 0.0,
            latest_frame: SonificationFrame::silent(),
        }
    }
}

pub(super) fn are_custom_clamps_valid(clamps: &[[f64; 2]]) -> bool {
    clamps.len() <= MAX_CUSTOM_CLAMPS
        && clamps.iter().all(|&[nx, ny]| {
            nx.is_finite()
                && (CLAMP_COORD_MIN..=CLAMP_COORD_MAX).contains(&nx)
                && ny.is_finite()
                && (CLAMP_COORD_MIN..=CLAMP_COORD_MAX).contains(&ny)
        })
}

pub(super) fn sanitize_custom_clamps(clamps: &mut Vec<[f64; 2]>) {
    clamps.truncate(MAX_CUSTOM_CLAMPS);
    for [nx, ny] in clamps.iter_mut() {
        *nx = if nx.is_finite() {
            nx.clamp(CLAMP_COORD_MIN, CLAMP_COORD_MAX)
        } else {
            0.5
        };
        *ny = if ny.is_finite() {
            ny.clamp(CLAMP_COORD_MIN, CLAMP_COORD_MAX)
        } else {
            0.5
        };
    }
}

impl ChladniWaveTankTool {
    pub(super) fn trigger_mallet_pulse(
        &mut self,
        nx: f64,
        ny: f64,
        sigma: f64,
        amplitude: f64,
        impulse: f32,
    ) {
        self.solver.inject_gaussian_pulse(nx, ny, sigma, amplitude);
        if impulse.is_finite() {
            self.audio.pending_impulse = self.audio.pending_impulse.max(impulse.clamp(0.0, 1.0));
        }
        self.refresh_analysis_outputs();
    }

    pub(super) fn update_sonification_frame(&mut self) {
        let field_finite = self.solver.displacement().iter().all(|v| v.is_finite());
        if !field_finite {
            self.audio.mode_count = 0;
            self.audio.pending_impulse = 0.0;
            self.audio.latest_frame = SonificationFrame::silent();
            crate::audio::notify_solver_diverged();
            return;
        }
        self.audio.mode_count = self.fft.extract_top_modes(&mut self.audio.top_modes);
        let base_len = if self.solver.dx() >= 0.5 {
            1.0
        } else {
            (((GRID_SIZE - 1) as f64) * self.solver.dx()).max(1.0e-3)
        };
        let preset_pins = match self.state.clamp_layout {
            super::ClampLayout::None => 0_usize,
            super::ClampLayout::CenterPin => 1,
            super::ClampLayout::QuadSymmetry | super::ClampLayout::EdgeMidpoints => 4,
        };
        let total_pins = preset_pins + self.state.custom_clamps.len();
        let domain_len = base_len / (1.0 + 0.035 * (total_pins as f64)).sqrt();
        let params = self.pde_params();
        self.audio.latest_frame = chladni_wave_tank_frame(
            &self.audio.top_modes[..self.audio.mode_count],
            &params,
            domain_len,
            &self.detector_buf,
            self.audio.pending_impulse,
        );
    }

    pub(super) fn draw_sonification_section(&mut self, ui: &mut egui::Ui) {
        render_sonification_panel(
            ui,
            SONIFICATION_TOOL_ID,
            &self.audio.latest_frame,
            self.paused,
        );
        if self.audio.pending_impulse > 0.0 {
            self.audio.pending_impulse = 0.0;
            for voice in &mut self.audio.latest_frame.voices {
                voice.transient_impulse = 0.0;
            }
        }
    }

    pub(super) fn try_handle_pickup_drag(&mut self, response: &egui::Response, rect: Rect) -> bool {
        if !response.dragged() {
            self.ir_studio.active_drag = PickupDragTarget::None;
            return false;
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return false;
        };
        let to_screen = |p: [f64; 2]| -> Pos2 {
            Pos2::new(
                rect.left() + (p[0] as f32) * rect.width(),
                rect.bottom() - (p[1] as f32) * rect.height(),
            )
        };
        if response.drag_started() || self.ir_studio.active_drag == PickupDragTarget::None {
            let origin = response
                .ctx
                .input(|i| i.pointer.press_origin())
                .unwrap_or(pos);
            let dl = origin.distance(to_screen(self.state.pickup_left));
            let dr = origin.distance(to_screen(self.state.pickup_right));
            self.ir_studio.active_drag = if dl <= 14.0_f32 && dl <= dr {
                PickupDragTarget::Left
            } else if dr <= 14.0_f32 {
                PickupDragTarget::Right
            } else {
                PickupDragTarget::None
            };
        }
        if self.ir_studio.active_drag == PickupDragTarget::None {
            return false;
        }
        let nx = (((pos.x - rect.left()) / rect.width()) as f64)
            .clamp(PICKUP_COORD_MIN, PICKUP_COORD_MAX);
        let ny = ((1.0 - (pos.y - rect.top()) / rect.height()) as f64)
            .clamp(PICKUP_COORD_MIN, PICKUP_COORD_MAX);
        match self.ir_studio.active_drag {
            PickupDragTarget::Left => self.state.pickup_left = [nx, ny],
            PickupDragTarget::Right => self.state.pickup_right = [nx, ny],
            PickupDragTarget::None => {}
        }
        self.state_rev = self.state_rev.wrapping_add(1);
        true
    }

    pub(super) fn paint_stereo_pickups(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        palette: SemanticPalette,
    ) {
        for (coord, label, color) in [
            (self.state.pickup_left, "L", palette.series_color(0)),
            (self.state.pickup_right, "R", palette.series_color(3)),
        ] {
            let center = Pos2::new(
                rect.left() + (coord[0] as f32) * rect.width(),
                rect.bottom() - (coord[1] as f32) * rect.height(),
            );
            painter.circle_filled(center, 7.5_f32, palette.surface_fill.gamma_multiply(0.9));
            painter.circle_stroke(center, 7.5_f32, Stroke::new(2.0_f32, color));
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::proportional(10.5_f32),
                color,
            );
        }
    }

    pub(super) fn draw_ir_studio_section(&mut self, ui: &mut egui::Ui) {
        ui.label("Stereo Impulse Response (.wav):");
        let mut changed = self.draw_ir_pickup_drags(ui);
        ui.horizontal(|ui| {
            ui.label("Decay (s):");
            changed |= ui
                .add(crate::framework::bounded_drag_value(
                    &mut self.state.ir_duration_secs,
                    IR_DURATION_MIN_SECS..=IR_DURATION_MAX_SECS,
                    0.05,
                ))
                .changed();
            egui::ComboBox::from_id_salt("chladni_ir_sr")
                .selected_text(self.state.ir_sample_rate.label())
                .width(76.0)
                .show_ui(ui, |ui| {
                    for r in WavSampleRate::ALL {
                        changed |= ui
                            .selectable_value(&mut self.state.ir_sample_rate, r, r.label())
                            .changed();
                    }
                });
        });
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("chladni_ir_fmt")
                .selected_text(self.state.ir_format.label())
                .width(98.0)
                .show_ui(ui, |ui| {
                    for f in WavSampleFormat::ALL {
                        changed |= ui
                            .selectable_value(&mut self.state.ir_format, f, f.label())
                            .changed();
                    }
                });
            if ui.button("🔔 Strike IR").clicked() {
                self.audition_strike_from_user_gesture();
            }
            if ui
                .add_enabled(
                    !self.ir_studio.is_rendering,
                    egui::Button::new("💾 Export IR"),
                )
                .clicked()
            {
                self.start_cooperative_ir_export(true);
            }
        });
        if self.ir_studio.is_rendering && self.ir_studio.render_total > 0 {
            let frac = (self.ir_studio.render_cursor as f32) / (self.ir_studio.render_total as f32);
            ui.add(egui::ProgressBar::new(frac).text(format!(
                "Rendering IR {:.0}% ({:.1} ms)",
                frac * 100.0,
                self.ir_studio.last_slice_ms
            )));
        }
        if changed {
            self.state_rev = self.state_rev.wrapping_add(1);
        }
    }

    fn draw_ir_pickup_drags(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Mic L:");
            changed |= ui
                .add(crate::framework::bounded_drag_value(
                    &mut self.state.pickup_left[0],
                    PICKUP_COORD_MIN..=PICKUP_COORD_MAX,
                    0.01,
                ))
                .changed();
            changed |= ui
                .add(crate::framework::bounded_drag_value(
                    &mut self.state.pickup_left[1],
                    PICKUP_COORD_MIN..=PICKUP_COORD_MAX,
                    0.01,
                ))
                .changed();
            ui.label("R:");
            changed |= ui
                .add(crate::framework::bounded_drag_value(
                    &mut self.state.pickup_right[0],
                    PICKUP_COORD_MIN..=PICKUP_COORD_MAX,
                    0.01,
                ))
                .changed();
            changed |= ui
                .add(crate::framework::bounded_drag_value(
                    &mut self.state.pickup_right[1],
                    PICKUP_COORD_MIN..=PICKUP_COORD_MAX,
                    0.01,
                ))
                .changed();
        });
        changed
    }
}
