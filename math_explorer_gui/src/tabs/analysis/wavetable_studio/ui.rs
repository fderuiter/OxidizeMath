#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:graph_parameters_rust]

use eframe::egui;
use egui_plot::{ExportPrimitive, Line, Plot, PlotPoints, VLine};
use math_explorer::pure_math::analysis::pde::wave_2d::{BoundaryMode2D, Wave2DMedium};

use super::{
    WavetableSourceMode, WavetableStudioTool, AUDITION_FREQ_MAX_HZ, AUDITION_FREQ_MIN_HZ,
    PDE_ANGLE_MAX_DEG, PDE_ANGLE_MIN_DEG, PDE_SPEED_MAX, PDE_SPEED_MIN, PDE_STIFFNESS_MAX,
    PDE_STIFFNESS_MIN,
};
use crate::accessibility::announce_status;
use crate::app::{palette::DARK_PALETTE, SemanticPalette};
use crate::audio::studio_preview::{
    snapshot_studio_preview, toggle_studio_wavetable_audition_from_gesture,
    update_studio_wavetable_audition,
};
use crate::audio::wav::{WavSampleFormat, WavSampleRate};
use crate::audio::wavetable::{MAX_WAVETABLE_FRAMES, WAVETABLE_FRAME_SAMPLES};
use crate::framework::bounded_drag_value;
use crate::reflective_ui::render_copyable_metric;

impl WavetableStudioTool {
    pub(super) fn draw_controls_sidebar(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "wavetable_studio_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Wavetable Studio");
                ui.separator();
                self.draw_source_mode_section(ui);
                ui.separator();
                self.draw_morph_and_bank_controls(ui);
                ui.separator();
                self.draw_audition_and_export_controls(ui);
                ui.separator();
                self.draw_spectral_metrics(ui);
            });
        });
    }

    fn draw_source_mode_section(&mut self, ui: &mut egui::Ui) {
        ui.label("Synthesis Source:");
        let prev_mode = self.state.source_mode;
        egui::ComboBox::from_id_salt("wt_source_mode")
            .selected_text(self.state.source_mode.label())
            .show_ui(ui, |ui| {
                for mode in [
                    WavetableSourceMode::FormulaExpr,
                    WavetableSourceMode::PdeWaveSlice,
                ] {
                    ui.selectable_value(&mut self.state.source_mode, mode, mode.label());
                }
            });
        if self.state.source_mode != prev_mode {
            self.pde_dirty = true;
            self.preview_dirty = true;
            self.bank_dirty = true;
            self.state_rev = self.state_rev.wrapping_add(1);
        }

        ui.add_space(4.0);
        match self.state.source_mode {
            WavetableSourceMode::FormulaExpr => {
                if self.expr_bar.ui(ui) {
                    self.sync_state_params_from_bar();
                    self.preview_dirty = true;
                    self.bank_dirty = true;
                    self.state_rev = self.state_rev.wrapping_add(1);
                }
            }
            WavetableSourceMode::PdeWaveSlice => self.draw_pde_slice_controls(ui),
        }
    }

    fn draw_pde_slice_controls(&mut self, ui: &mut egui::Ui) {
        let mut pde_changed = false;
        let prev_med = self.state.pde_medium;
        egui::ComboBox::from_id_salt("wt_pde_medium")
            .selected_text(format!("{:?}", self.state.pde_medium))
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut self.state.pde_medium,
                    Wave2DMedium::BiharmonicPlate,
                    "Biharmonic Plate",
                );
                ui.selectable_value(
                    &mut self.state.pde_medium,
                    Wave2DMedium::AcousticWave,
                    "Acoustic Wave",
                );
            });
        pde_changed |= self.state.pde_medium != prev_med;

        let prev_bnd = self.state.pde_boundary;
        egui::ComboBox::from_id_salt("wt_pde_boundary")
            .selected_text(format!("{:?}", self.state.pde_boundary))
            .show_ui(ui, |ui| {
                for b in [
                    BoundaryMode2D::Clamped,
                    BoundaryMode2D::Dirichlet,
                    BoundaryMode2D::Neumann,
                ] {
                    ui.selectable_value(&mut self.state.pde_boundary, b, format!("{b:?}"));
                }
            });
        pde_changed |= self.state.pde_boundary != prev_bnd;
        pde_changed |= self.draw_pde_numeric_drags(ui);
        if pde_changed {
            self.pde_dirty = true;
            self.preview_dirty = true;
            self.bank_dirty = true;
            self.state_rev = self.state_rev.wrapping_add(1);
        }
    }

    fn draw_pde_numeric_drags(&mut self, ui: &mut egui::Ui) -> bool {
        let mut pde_changed = false;
        if self.state.pde_medium == Wave2DMedium::BiharmonicPlate {
            ui.horizontal(|ui| {
                ui.label("Stiffness D:");
                pde_changed |= ui
                    .add(bounded_drag_value(
                        &mut self.state.pde_stiffness,
                        PDE_STIFFNESS_MIN..=PDE_STIFFNESS_MAX,
                        1.0e-5,
                    ))
                    .changed();
            });
        } else {
            ui.horizontal(|ui| {
                ui.label("Wave Speed c:");
                pde_changed |= ui
                    .add(bounded_drag_value(
                        &mut self.state.pde_wave_speed,
                        PDE_SPEED_MIN..=PDE_SPEED_MAX,
                        0.05,
                    ))
                    .changed();
            });
        }
        ui.horizontal(|ui| {
            ui.label("Slice Angle (°):");
            if ui
                .add(bounded_drag_value(
                    &mut self.state.pde_slice_angle_deg,
                    PDE_ANGLE_MIN_DEG..=PDE_ANGLE_MAX_DEG,
                    1.0,
                ))
                .changed()
            {
                self.preview_dirty = true;
                self.bank_dirty = true;
                self.state_rev = self.state_rev.wrapping_add(1);
            }
        });
        pde_changed
    }

    fn draw_morph_and_bank_controls(&mut self, ui: &mut egui::Ui) {
        let mut morph_changed = false;
        if ui
            .add(egui::Slider::new(&mut self.state.morph_pos, 0.0..=1.0).text("Morph m"))
            .changed()
        {
            morph_changed = true;
        }
        let max_f = MAX_WAVETABLE_FRAMES as u16;
        if ui
            .add(egui::Slider::new(&mut self.state.frame_count, 1..=max_f).text("Frames"))
            .changed()
        {
            self.bank_dirty = true;
            self.state_rev = self.state_rev.wrapping_add(1);
        }
        ui.horizontal(|ui| {
            ui.label("Band-limit f0:");
            if ui
                .add(bounded_drag_value(
                    &mut self.state.bandlimit_fundamental_hz,
                    AUDITION_FREQ_MIN_HZ..=AUDITION_FREQ_MAX_HZ,
                    1.0,
                ))
                .changed()
            {
                morph_changed = true;
                self.bank_dirty = true;
            }
        });
        if morph_changed {
            self.state_rev = self.state_rev.wrapping_add(1);
            self.refresh_active_preview();
            let _ = update_studio_wavetable_audition(
                &self.engine.periodic_real,
                &self.engine.periodic_imag,
                self.state.audition_freq_hz as f32,
                0.70,
            );
        }
    }

    fn draw_audition_and_export_controls(&mut self, ui: &mut egui::Ui) {
        let mut fmt_changed = false;
        ui.horizontal(|ui| {
            ui.label("Pitch (Hz):");
            if ui
                .add(bounded_drag_value(
                    &mut self.state.audition_freq_hz,
                    AUDITION_FREQ_MIN_HZ..=AUDITION_FREQ_MAX_HZ,
                    1.0,
                ))
                .changed()
            {
                fmt_changed = true;
                let _ = update_studio_wavetable_audition(
                    &self.engine.periodic_real,
                    &self.engine.periodic_imag,
                    self.state.audition_freq_hz as f32,
                    0.70,
                );
            }
        });
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("wt_sr")
                .selected_text(self.state.sample_rate.label())
                .width(82.0)
                .show_ui(ui, |ui| {
                    for r in WavSampleRate::ALL {
                        fmt_changed |= ui
                            .selectable_value(&mut self.state.sample_rate, r, r.label())
                            .changed();
                    }
                });
            egui::ComboBox::from_id_salt("wt_fmt")
                .selected_text(self.state.sample_format.label())
                .width(102.0)
                .show_ui(ui, |ui| {
                    for f in WavSampleFormat::ALL {
                        fmt_changed |= ui
                            .selectable_value(&mut self.state.sample_format, f, f.label())
                            .changed();
                    }
                });
        });
        if fmt_changed {
            self.preview_dirty = true;
            self.bank_dirty = true;
            self.state_rev = self.state_rev.wrapping_add(1);
        }
        self.draw_audition_and_export_buttons(ui);
    }

    fn draw_audition_and_export_buttons(&mut self, ui: &mut egui::Ui) {
        let snap = snapshot_studio_preview();
        let aud_label = if snap.wavetable_auditioning {
            "⏹ Stop Audition"
        } else {
            "🔊 Audition Morph"
        };
        ui.horizontal_wrapped(|ui| {
            if ui.button(aud_label).clicked() {
                self.toggle_audition_from_gesture();
            }
            if ui.button("💾 Export .wav Bank").clicked() {
                self.export_wavetable_bank_file();
            }
        });
        ui.small("Exports 2048-sample Vital / Serum / Surge XT bank (clm + uhWT chunks).");
    }

    pub(crate) fn toggle_audition_from_gesture(&mut self) {
        if self.preview_dirty {
            self.refresh_active_preview();
        }
        match toggle_studio_wavetable_audition_from_gesture(
            &self.engine.periodic_real,
            &self.engine.periodic_imag,
            self.state.audition_freq_hz as f32,
            0.70,
        ) {
            Ok(true) => announce_status("Wavetable PeriodicWave audition active."),
            Ok(false) => announce_status("Wavetable audition stopped."),
            Err(err) => announce_status(&format!("Wavetable audition unavailable: {err}")),
        }
    }

    pub(crate) fn export_wavetable_bank_file(&mut self) {
        let wav_bytes = self.synthesize_and_encode_wav();
        let filename = format!(
            "oxidize_wavetable_{}x2048_{}hz.wav",
            self.state.frame_count,
            self.state.sample_rate.hz()
        );
        #[cfg(test)]
        {
            self.last_exported_wav = Some(wav_bytes.clone());
        }
        #[cfg(not(all(test, not(target_arch = "wasm32"))))]
        {
            let (tx, _rx) = std::sync::mpsc::channel();
            let config = crate::web_io::SaveAsDialogConfig {
                filter_name: "Synthesizer Wavetable (*.wav)",
                extensions: &["wav"],
                fallback_filename: "oxidize_wavetable.wav",
                write_error_prefix: "Failed to write wavetable WAV",
            };
            crate::web_io::spawn_save_binary_file_dialog(tx, config, &filename, wav_bytes);
        }
        announce_status(&format!("Exported {filename} wavetable bank."));
    }

    fn draw_spectral_metrics(&self, ui: &mut egui::Ui) {
        render_copyable_metric(
            ui,
            "Bank Payload",
            &format!(
                "{} × {} samples",
                self.state.frame_count, WAVETABLE_FRAME_SAMPLES
            ),
        );
        render_copyable_metric(
            ui,
            "Nyquist Cutoff",
            &format!("Harmonic k <= {}", self.engine.last_nyquist_bin),
        );
        render_copyable_metric(
            ui,
            "Frame DC Mean",
            &format!("{:.2e}", self.engine.last_dc_mean),
        );
    }

    pub(super) fn draw_central_plots(&self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading("PDE & Formula Wavetable Studio");
            let palette = SemanticPalette::from_visuals(ui.visuals());
            if ui.available_width() > 640.0 {
                ui.columns(2, |cols| {
                    self.draw_single_cycle_plot(&mut cols[0], palette);
                    self.draw_harmonic_spectrum_plot(&mut cols[1], palette);
                });
            } else {
                self.draw_single_cycle_plot(ui, palette);
                ui.add_space(6.0);
                self.draw_harmonic_spectrum_plot(ui, palette);
            }
        });
    }

    fn draw_single_cycle_plot(&self, ui: &mut egui::Ui, palette: SemanticPalette) {
        ui.label(format!(
            "Single-Cycle Frame (2048 samples, morph m = {:.2})",
            self.state.morph_pos
        ));
        let to_pts = |arr: &[f32; WAVETABLE_FRAME_SAMPLES], step: usize| -> PlotPoints {
            arr.iter()
                .enumerate()
                .step_by(step)
                .map(|(i, &v)| {
                    let x = -1.0 + 2.0 * (i as f64) / (WAVETABLE_FRAME_SAMPLES as f64);
                    [x, if v.is_finite() { v as f64 } else { 0.0 }]
                })
                .collect()
        };
        let ghost_start = to_pts(&self.engine.keyframe_start, 4);
        let ghost_end = to_pts(&self.engine.keyframe_end, 4);
        let active = to_pts(&self.engine.active_frame, 2);

        Plot::new("wavetable_single_cycle_plot")
            .allow_boxed_zoom(false)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("m = 0.0", ghost_start)
                        .color(palette.series_color(2).gamma_multiply(0.40))
                        .width(1.1_f32),
                );
                plot_ui.line(
                    Line::new("m = 1.0", ghost_end)
                        .color(palette.series_color(3).gamma_multiply(0.40))
                        .width(1.1_f32),
                );
                plot_ui.line(
                    Line::new("Active Frame", active)
                        .color(palette.series_color(0))
                        .width(2.2_f32),
                );
            });
    }

    fn draw_harmonic_spectrum_plot(&self, ui: &mut egui::Ui, palette: SemanticPalette) {
        ui.label("Band-Limited Harmonic Partials |c_k| (k = 1..64)");
        let pts: PlotPoints = self
            .engine
            .harmonic_mags
            .iter()
            .enumerate()
            .take(64)
            .map(|(idx, &m)| [(idx + 1) as f64, if m.is_finite() { m as f64 } else { 0.0 }])
            .collect();
        let cutoff = (self.engine.last_nyquist_bin.min(64)) as f64;
        Plot::new("wavetable_harmonic_plot")
            .allow_boxed_zoom(false)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Harmonic |c_k|", pts)
                        .color(palette.series_color(1))
                        .width(2.0_f32),
                );
                plot_ui.vline(VLine::new("Nyquist Limit", cutoff).color(palette.warning));
            });
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let pts: Vec<[f64; 2]> = self
            .engine
            .active_frame
            .iter()
            .enumerate()
            .step_by(2)
            .map(|(i, &s)| {
                let x = -1.0 + 2.0 * (i as f64) / (WAVETABLE_FRAME_SAMPLES as f64);
                [x, if s.is_finite() { s as f64 } else { 0.0 }]
            })
            .collect();
        vec![ExportPrimitive::Polyline {
            name: "Wavetable Single-Cycle (2048)".into(),
            points: pts.into(),
            color: DARK_PALETTE.series_color(0),
            stroke_width: 2.0,
            closed: false,
            fill: None,
        }]
    }
}
