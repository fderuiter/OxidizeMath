#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel controls for [`super::state::FractranMachineTool`].

use super::state::{FractranMachineTool, FractranPreset, MAX_INITIAL_VALUE, MAX_STEPS_PER_FRAME};
use crate::accessibility::AccessibleHoverText;
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui;
use math_explorer::pure_math::algorithmic_information::fractran::{
    FractranProgram, FIRST_64_PRIMES,
};

impl FractranMachineTool {
    pub(crate) fn select_preset(&mut self, preset: FractranPreset) {
        self.config.preset = preset;
        if preset != FractranPreset::Custom {
            self.config.custom_fractions = preset.canonical_fractions_text().to_string();
            self.config.initial_value = preset.default_initial_value();
            self.config.warmup_steps = if preset == FractranPreset::Primegame {
                24
            } else {
                0
            };
        }
        self.rebuild_machine(self.config.warmup_steps);
    }

    fn show_preset_selector(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Canonical Programs").strong());
        ui.horizontal_wrapped(|ui| {
            for preset in [
                FractranPreset::Primegame,
                FractranPreset::Fibonacci,
                FractranPreset::Multiplier,
                FractranPreset::Adder,
                FractranPreset::Custom,
            ] {
                let selected = self.config.preset == preset;
                if ui
                    .selectable_label(selected, preset.label())
                    .accessible_hover_text(format!("Load {} FRACTRAN program", preset.label()))
                    .clicked()
                {
                    self.select_preset(preset);
                }
            }
        });
    }

    fn show_transport_controls(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Execution Transport").strong());
        if let Some(action) = PlaybackTransport::in_thread(
            "fractran_transport",
            "FRACTRAN Virtual Machine",
            self.is_playing,
        )
        .show(ui)
        {
            match action {
                PlaybackAction::Play => {
                    if !self.live.halted && self.live.cycle.is_none() {
                        self.is_playing = true;
                        self.status_banner = None;
                    }
                    self.bump_revision();
                }
                PlaybackAction::Pause => {
                    self.is_playing = false;
                    self.bump_revision();
                }
                PlaybackAction::Step => {
                    self.is_playing = false;
                    let _ = self.step_once();
                }
                PlaybackAction::Reset => {
                    self.config.warmup_steps = 0;
                    self.rebuild_machine(0);
                }
            }
        }

        let mut spf = self.config.steps_per_frame;
        if ui
            .add(egui::Slider::new(&mut spf, 1..=MAX_STEPS_PER_FRAME).text("Steps / Frame"))
            .accessible_hover_text("Maximum transitions executed per UI frame (capped at 256)")
            .changed()
        {
            self.config.steps_per_frame = spf.clamp(1, MAX_STEPS_PER_FRAME);
            self.bump_revision();
        }

        let mut rate = self.config.playback_rate;
        if ui
            .add(egui::Slider::new(&mut rate, 0.25..=4.0).text("Rate Scale"))
            .accessible_hover_text("Playback rate multiplier applied to steps per frame")
            .changed()
        {
            self.config.playback_rate = rate.clamp(0.25, 16.0);
            self.bump_revision();
        }
    }

    fn show_initial_state_editor(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Initial Register Integer N0").strong());
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.initial_input_text)
                    .desired_width(120.0)
                    .hint_text("e.g. 2 or 216"),
            );
            if ui
                .button("Apply N0")
                .accessible_hover_text("Factor positive integer N0 into prime registers p <= 311")
                .clicked()
            {
                self.apply_initial_input_text();
            }
        });
    }

    pub(crate) fn apply_initial_input_text(&mut self) {
        match self.initial_input_text.trim().parse::<u64>() {
            Ok(val) if (1..=MAX_INITIAL_VALUE).contains(&val) => {
                match FractranProgram::encode_u64(val) {
                    Ok(_) => {
                        self.config.initial_value = val;
                        self.config.warmup_steps = 0;
                        self.validation_error = None;
                        self.rebuild_machine(0);
                    }
                    Err(err) => {
                        self.validation_error = Some(err.to_string());
                    }
                }
            }
            _ => {
                self.validation_error = Some(format!(
                    "Initial integer must be within 1..={MAX_INITIAL_VALUE}"
                ));
            }
        }
    }

    fn show_custom_fractions_editor(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Fraction Rules (num/den)").strong());
        let edited = ui
            .add(
                egui::TextEdit::multiline(&mut self.config.custom_fractions)
                    .desired_rows(4)
                    .hint_text("e.g. 17/91, 78/85, 55/1"),
            )
            .changed();
        if edited {
            self.config.preset = FractranPreset::Custom;
            self.bump_revision();
        }
        if ui
            .button("Compile Fraction Rules")
            .accessible_hover_text("Validate and pre-factor custom fraction rules over p <= 311")
            .clicked()
        {
            self.compile_custom_rules();
        }
    }

    pub(crate) fn compile_custom_rules(&mut self) {
        self.config.preset = FractranPreset::Custom;
        match FractranProgram::parse(&self.config.custom_fractions) {
            Ok(program) => {
                self.program = program;
                self.config.warmup_steps = 0;
                self.validation_error = None;
                self.rebuild_machine(0);
            }
            Err(err) => {
                self.is_playing = false;
                self.validation_error = Some(err.to_string());
                self.bump_revision();
            }
        }
    }

    fn show_telemetry_card(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.group(|ui| {
            ui.label(egui::RichText::new("Machine Telemetry").strong());
            ui.label(format!("Transitions: {}", self.live.steps));
            ui.label(format!(
                "Approx State Size: {:.1} bits",
                self.estimated_state_bit_length()
            ));
            ui.label(format!(
                "Dyadic Point K_12(r2, r3): {:.2} bits",
                self.dyadic_demo_k
            ));
            let state_label = if self.live.cycle.is_some() {
                ("Cycle Paused (Floyd)", palette.warning)
            } else if self.live.halted {
                ("Halted", palette.info)
            } else if self.is_playing {
                ("Running", palette.success)
            } else {
                ("Ready", ui.visuals().text_color())
            };
            ui.colored_label(state_label.1, format!("Status: {}", state_label.0));
        });

        if let Some(banner) = &self.status_banner {
            ui.colored_label(palette.warning, banner);
        }
        if let Some(err) = &self.validation_error {
            ui.colored_label(palette.error, err);
        }
    }

    pub(super) fn stop_live_playback_and_mute(&mut self) {
        if std::mem::take(&mut self.is_playing) {
            crate::audio::notify_playback_paused();
        }
    }

    /// Returns `true` when the machine has halted, entered a Floyd cycle, or encountered a validation error.
    #[must_use]
    pub fn is_audio_silenced(&self) -> bool {
        self.live.halted || self.live.cycle.is_some() || self.validation_error.is_some()
    }

    /// Extracts the current zero-allocation `just_intonation_ratio` chord and Kolmogorov timbre frame.
    #[must_use]
    pub fn current_sonification_frame(&self) -> crate::audio::SonificationFrame {
        let pow2_hit = if self.live.steps > 0 {
            FractranProgram::power_of_two_exponent(&self.live.exponents)
        } else {
            None
        };
        let (bdm_bits, lz76_bits) = self
            .trajectory
            .last()
            .map_or((0.0, 0.0), |pt| (pt.bdm_bits, pt.lz76_bits));
        crate::audio::mappings::fractran_register_frame(
            &FIRST_64_PRIMES,
            &self.live.exponents,
            pow2_hit,
            bdm_bits,
            lz76_bits,
            self.is_audio_silenced(),
        )
    }

    pub(crate) fn show_controls_panel(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("FRACTRAN Virtual Machine");
            ui.separator();
            self.show_preset_selector(ui);
            ui.separator();
            self.show_transport_controls(ui);
            ui.separator();
            let frame = self.current_sonification_frame();
            crate::audio::render_sonification_panel(
                ui,
                "algorithmic_fractran",
                &frame,
                self.is_audio_silenced(),
            );
            ui.separator();
            self.show_initial_state_editor(ui);
            ui.separator();
            self.show_custom_fractions_editor(ui);
            ui.separator();
            self.show_telemetry_card(ui);
        });
    }
}

// [cite:algorithmic_information_rust]
