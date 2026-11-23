#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel optical parameter controls, playback transport, and invariant telemetry badges.

// [cite:graph_parameters_rust]

use eframe::egui;
use math_explorer::physics::optics::holography::HologramTargetPreset;

use super::{
    AperturePreset, FourierOpticsWorkbenchTool, OpticsDisplayMode, OpticsMode,
    REAL_SCALAR_TWIN_FLOOR,
};
use crate::app::SemanticPalette;
use crate::framework::{
    PaneSidePanel, PlaybackAction, PlaybackTransport, bounded_drag_value, scoped_panel_id,
};
use crate::reflective_ui::render_copyable_metric;
use crate::web_io::PlatformInstant;

impl FourierOpticsWorkbenchTool {
    pub(super) fn render_workbench(&mut self, parent_ui: &mut egui::Ui) {
        self.ensure_evaluated();
        let ctx = parent_ui.ctx().clone();
        if self.state.is_playing {
            let now = PlatformInstant::now();
            if now.duration_since(self.last_tick).as_millis() >= 32 {
                self.last_tick = now;
                self.advance_playback_step();
            }
            ctx.request_repaint();
        }
        self.draw_side_controls(parent_ui, &ctx);
        self.draw_central_canvas_and_plots(parent_ui);
    }

    pub(crate) fn advance_playback_step(&mut self) {
        match self.state.mode {
            OpticsMode::HelmholtzDiffraction => {
                let next = self.state.propagation_mm + 0.4;
                self.state.propagation_mm = if next > self.state.carpet_max_mm {
                    0.2
                } else {
                    next
                };
            }
            OpticsMode::GerchbergSaxtonHolography => {
                self.state.gs_iterations = if self.state.gs_iterations >= 80 {
                    1
                } else {
                    self.state.gs_iterations + 1
                };
            }
        }
        self.clamp_state_in_place();
        self.revision = self.revision.wrapping_add(1);
    }

    fn draw_side_controls(&mut self, parent_ui: &mut egui::Ui, ctx: &egui::Context) {
        let panel_id = scoped_panel_id(parent_ui, "fourier_optics_side_panel");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Fourier Optics Workbench");
                ui.separator();
                let mut dirty = self.draw_mode_and_view_selectors(ui);
                ui.separator();
                dirty |= self.draw_beam_and_grid_controls(ui);
                ui.separator();
                if self.state.mode == OpticsMode::HelmholtzDiffraction {
                    dirty |= self.draw_diffraction_controls(ui);
                } else {
                    dirty |= self.draw_holography_controls(ui);
                }
                if dirty {
                    self.clamp_state_in_place();
                    self.revision = self.revision.wrapping_add(1);
                }
                ui.separator();
                self.draw_playback_controls(ui, ctx);
                ui.separator();
                self.draw_invariant_telemetry(ui);
            });
        });
    }

    fn draw_mode_and_view_selectors(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::ComboBox::from_label("Engine Mode")
            .selected_text(self.state.mode.label())
            .show_ui(ui, |ui| {
                for m in [
                    OpticsMode::HelmholtzDiffraction,
                    OpticsMode::GerchbergSaxtonHolography,
                ] {
                    changed |= ui
                        .selectable_value(&mut self.state.mode, m, m.label())
                        .changed();
                }
            });

        egui::ComboBox::from_label("Viewport Layer")
            .selected_text(self.state.display.label())
            .show_ui(ui, |ui| {
                for d in [
                    OpticsDisplayMode::IntensityXY,
                    OpticsDisplayMode::WrappedPhaseXY,
                    OpticsDisplayMode::AxialCarpetXZ,
                    OpticsDisplayMode::SlmPhaseMask8Bit,
                ] {
                    changed |= ui
                        .selectable_value(&mut self.state.display, d, d.label())
                        .changed();
                }
            });

        changed |= ui
            .checkbox(
                &mut self.state.high_contrast_colormap,
                "High-Contrast Perceptual Colormap",
            )
            .changed();
        changed
    }

    fn draw_beam_and_grid_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Wavelength (nm):");
            changed |= ui
                .add(bounded_drag_value(
                    &mut self.state.wavelength_nm,
                    380.0..=1550.0,
                    2.0,
                ))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Pixel Pitch (um):");
            changed |= ui
                .add(bounded_drag_value(
                    &mut self.state.pixel_pitch_um,
                    2.0..=25.0,
                    0.2,
                ))
                .changed();
        });
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.propagation_mm, -150.0..=150.0)
                    .text("Propagation z (mm)"),
            )
            .changed();
        changed
    }

    fn draw_diffraction_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::ComboBox::from_label("Aperture Mask")
            .selected_text(self.state.aperture.label())
            .show_ui(ui, |ui| {
                for ap in [
                    AperturePreset::OpaqueCircularDisk,
                    AperturePreset::SingleMultiSlit,
                    AperturePreset::RonchiGrating,
                    AperturePreset::ThinQuadraticLens,
                ] {
                    changed |= ui
                        .selectable_value(&mut self.state.aperture, ap, ap.label())
                        .changed();
                }
            });

        let scale_label = match self.state.aperture {
            AperturePreset::OpaqueCircularDisk => "Disk Radius a (um)",
            AperturePreset::SingleMultiSlit => "Slit Width w (um)",
            AperturePreset::RonchiGrating => "Grating Period d (um)",
            AperturePreset::ThinQuadraticLens => "Pupil Radius R (um)",
        };
        changed |= ui
            .add(egui::Slider::new(&mut self.state.feature_scale_um, 16.0..=320.0).text(scale_label))
            .changed();

        match self.state.aperture {
            AperturePreset::SingleMultiSlit => {
                changed |= ui
                    .add(egui::Slider::new(&mut self.state.slit_count, 1..=7).text("Slit Count N"))
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.state.slit_pitch_um, 16.0..=400.0)
                            .text("Slit Pitch (um)"),
                    )
                    .changed();
            }
            AperturePreset::RonchiGrating => {
                changed |= ui
                    .add(egui::Slider::new(&mut self.state.duty_cycle, 0.1..=0.9).text("Duty Cycle"))
                    .changed();
            }
            AperturePreset::ThinQuadraticLens => {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.state.focal_length_mm, 2.0..=120.0)
                            .text("Focal Length f (mm)"),
                    )
                    .changed();
            }
            AperturePreset::OpaqueCircularDisk => {}
        }

        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.carpet_max_mm, 1.0..=200.0)
                    .text("Carpet Range z_max (mm)"),
            )
            .changed();

        changed |= self.draw_zernike_controls(ui);
        changed
    }

    fn draw_zernike_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::CollapsingHeader::new("Zernike Wavefront Aberrations")
            .default_open(self.state.aperture == AperturePreset::ThinQuadraticLens)
            .show(ui, |ui| {
                changed |= ui
                    .add(egui::Slider::new(&mut self.state.zernike_n, 0..=6).text("Radial Order n"))
                    .changed();
                let n_i32 = self.state.zernike_n as i32;
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.state.zernike_m, -n_i32..=n_i32)
                            .text("Azimuthal Order m"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.state.zernike_rms_waves, -2.0..=2.0)
                            .text("RMS Aberration (waves)"),
                    )
                    .changed();
            });
        changed
    }

    fn draw_holography_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let target_label = |t: HologramTargetPreset| match t {
            HologramTargetPreset::OffAxisRingAndSpot => "Off-Axis Ring & Spot",
            HologramTargetPreset::AsymmetricReticle => "Asymmetric Quadrant Reticle",
            HologramTargetPreset::UpperTriadSpots => "Upper-Half Triad Spots",
            HologramTargetPreset::OffAxisGlyphF => "Off-Axis Letter F",
        };
        egui::ComboBox::from_label("Target Pattern")
            .selected_text(target_label(self.state.gs_target))
            .show_ui(ui, |ui| {
                for t in [
                    HologramTargetPreset::OffAxisRingAndSpot,
                    HologramTargetPreset::AsymmetricReticle,
                    HologramTargetPreset::UpperTriadSpots,
                    HologramTargetPreset::OffAxisGlyphF,
                ] {
                    changed |= ui
                        .selectable_value(&mut self.state.gs_target, t, target_label(t))
                        .changed();
                }
            });

        changed |= ui
            .checkbox(
                &mut self.state.gs_lensless,
                "Lensless Near-Field Angular Spectrum",
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.gs_iterations, 1..=100).text("GS Iterations"))
            .changed();
        if ui.button("Reseed OxidizeRng Phase").clicked() {
            self.state.gs_seed = self.state.gs_seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            changed = true;
        }
        changed
    }

    fn draw_playback_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if let Some(action) = PlaybackTransport::in_thread(
            "fourier_optics_transport",
            "Fourier Optics Workbench",
            self.state.is_playing,
        )
        .show(ui)
        {
            match action {
                PlaybackAction::Play => {
                    self.state.is_playing = true;
                    self.last_tick = PlatformInstant::now();
                }
                PlaybackAction::Pause => self.state.is_playing = false,
                PlaybackAction::Step => {
                    self.state.is_playing = false;
                    self.advance_playback_step();
                }
                PlaybackAction::Reset => {
                    self.state.is_playing = false;
                    if self.state.mode == OpticsMode::HelmholtzDiffraction {
                        self.state.propagation_mm = 16.384;
                    } else {
                        self.state.gs_iterations = 50;
                    }
                    self.revision = self.revision.wrapping_add(1);
                }
            }
            ctx.request_repaint();
        }
    }

    fn draw_invariant_telemetry(&self, ui: &mut egui::Ui) {
        self.ensure_evaluated();
        let guard = self.cache.borrow();
        let Some(c) = guard.as_ref() else {
            return;
        };
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.strong("Physical Invariants & Telemetry");
        match self.state.mode {
            OpticsMode::HelmholtzDiffraction => {
                let drift = if c.power_in > 1.0e-24 {
                    (c.power_out - c.power_in).abs() / c.power_in
                } else {
                    0.0
                };
                render_copyable_metric(ui, "Fresnel Number N_F", &format!("{:.4}", c.fresnel_nf))
                    .on_hover_text("Dimensionless Fresnel zone count a^2 / (lambda * |z|)");
                render_copyable_metric(ui, "Talbot Length z_T (mm)", &format!("{:.3}", c.talbot_mm))
                    .on_hover_text("Paraxial periodic self-imaging revival length 2 d^2 / lambda");
                render_copyable_metric(
                    ui,
                    "On-Axis Intensity I(0,0)",
                    &format!("{:.4}", c.center_intensity),
                )
                .on_hover_text("Normalized intensity on the optical axis (exceeds 1.0 for Arago spot at N_F = 2)");
                ui.colored_label(
                    palette.success,
                    format!("Parseval Power Drift: {drift:.2e}"),
                );
            }
            OpticsMode::GerchbergSaxtonHolography => {
                render_copyable_metric(
                    ui,
                    "Target Amplitude NRMSE",
                    &format!("{:.4}", c.gs_metrics.target_nrmse),
                )
                .on_hover_text("Scale-normalized root-mean-square error against target amplitude (< 0.32 beats 0.707 twin floor)");
                render_copyable_metric(
                    ui,
                    "Diffraction Efficiency",
                    &format!("{:.2}%", c.gs_metrics.diffraction_efficiency * 100.0),
                )
                .on_hover_text("Fraction of replay optical power concentrated inside target support");
                render_copyable_metric(
                    ui,
                    "Conjugate Twin Leakage",
                    &format!("{:.2}%", c.gs_metrics.twin_leakage * 100.0),
                )
                .on_hover_text("Residual power leaking into centrosymmetric conjugate twin region");
                ui.colored_label(
                    palette.info,
                    format!(
                        "Real-Scalar Twin Floor: {:.3} (measured {:.3})",
                        REAL_SCALAR_TWIN_FLOOR, c.real_scalar_metrics.target_nrmse
                    ),
                );
            }
        }
    }
}
