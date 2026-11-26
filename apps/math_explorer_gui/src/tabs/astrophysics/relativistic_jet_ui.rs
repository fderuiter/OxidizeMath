#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::relativistic_jet_tool::{JetPlotMode, RelativisticJetBeamingLab};
use crate::framework::TraceSeries;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, LineStyle, Plot, PlotPoints};
use math_explorer::physics::high_energy::evaluate_jet_knot_observables;

impl RelativisticJetBeamingLab {
    pub(super) fn render_observables_readouts(&self, ui: &mut egui::Ui) {
        let o = &self.observables;
        ui.heading("Beaming & Shock Readouts");
        render_copyable_metric(ui, "Bulk Lorentz Factor γ", &format!("{:.3}", o.bulk_gamma));
        render_copyable_metric(
            ui,
            "Doppler D_app / D_rec",
            &format!("{:.3} / {:.3}", o.doppler_approaching, o.doppler_receding),
        );
        render_copyable_metric(
            ui,
            "Apparent Speed β_app",
            &format!("{:.3} c", o.apparent_beta),
        );
        render_copyable_metric(
            ui,
            "Twin-Jet Flux Ratio",
            &format!("{:.1}x", o.jet_flux_asymmetry),
        );
        render_copyable_metric(
            ui,
            "EHT Disk Asymmetry",
            &format!("{:.2}x", o.disk_flux_asymmetry),
        );
        render_copyable_metric(
            ui,
            "Shock Compression r_sh",
            &format!("{:.2}x", o.shock_compression),
        );
        render_copyable_metric(
            ui,
            "Boosted Synchrotron P",
            &format!("{:.3e} W", o.boosted_synchrotron_power_w),
        );
        render_copyable_metric(
            ui,
            "Li & Ma Significance",
            &format!("{:.2} σ", o.li_ma_sigma),
        );
    }

    pub(super) fn render_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut ch = false;
        ui.heading("Jet & Disk Geometry");
        egui::ComboBox::from_label("Visualization Mode")
            .selected_text(self.cfg.plot_mode.label())
            .show_ui(ui, |ui| {
                for m in [
                    JetPlotMode::PolarTwinJetLobe,
                    JetPlotMode::AccretionRingProfile,
                ] {
                    ch |= ui
                        .selectable_value(&mut self.cfg.plot_mode, m, m.label())
                        .changed();
                }
            });
        ch |= self.render_kinematic_sliders(ui);
        ui.separator();
        ch |= self.render_shock_and_detection_sliders(ui);
        ch
    }

    fn render_kinematic_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let c = &mut self.cfg;
        let mut ch = false;
        ch |= ui
            .add(egui::Slider::new(&mut c.bulk_beta, 0.05..=0.995).text("Bulk Jet Speed β = v/c"))
            .changed();
        ch |= ui
            .add(
                egui::Slider::new(&mut c.viewing_angle_deg, 1.0..=89.0)
                    .text("Jet Viewing Angle θ (deg)"),
            )
            .changed();
        ch |= ui
            .add(
                egui::Slider::new(&mut c.disk_r_over_rs, 1.55..=20.0)
                    .text("Accretion Ring r / R_s"),
            )
            .changed();
        ch |= ui
            .add(
                egui::Slider::new(&mut c.disk_inclination_deg, 0.0..=90.0)
                    .text("Disk Inclination i (deg)"),
            )
            .changed();
        ch
    }

    fn render_shock_and_detection_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let c = &mut self.cfg;
        let mut ch = false;
        ui.heading("Shock, Synchrotron & Li-Ma");
        ch |= ui
            .add(
                egui::Slider::new(&mut c.electron_index_p, 1.2..=4.5)
                    .text("Electron Spectral Index p"),
            )
            .changed();
        ch |= ui
            .add(
                egui::Slider::new(&mut c.electron_log10_gamma, 1.0..=6.0)
                    .text("Electron log10(γ_e)"),
            )
            .changed();
        ch |= ui
            .add(
                egui::Slider::new(&mut c.upstream_log10_ub, -6.0..=2.0)
                    .text("Upstream log10(U_B [J/m^3])"),
            )
            .changed();
        ch |= ui
            .add(egui::Slider::new(&mut c.adiabatic_index, 1.1..=2.0).text("Adiabatic Index γ_ad"))
            .changed();
        ch |= ui
            .add(
                egui::Slider::new(&mut c.intrinsic_signal_counts, 0.01..=100.0)
                    .text("Comoving Signal Photons"),
            )
            .changed();
        ch |= ui
            .add(
                egui::Slider::new(&mut c.background_counts_off, 1.0..=5_000.0)
                    .text("Background Counts N_off"),
            )
            .changed();
        ch
    }

    pub(super) fn render_sensitivity_and_snapshots(&mut self, ui: &mut egui::Ui) {
        let base_cfg = self.cfg.clone();
        let params = [
            ("Bulk Jet Speed beta", base_cfg.bulk_beta),
            ("Viewing Angle (deg)", base_cfg.viewing_angle_deg),
            ("Electron Index p", base_cfg.electron_index_p),
            ("Adiabatic Index", base_cfg.adiabatic_index),
        ];
        let ctx_bits = crate::sensitivity::canonical_f64_bits(base_cfg.background_counts_off);
        self.sensitivity_engine
            .compute_with_context(&params, &ctx_bits, |p_map| {
                let mut probe = base_cfg.clone();
                if let Some(&b) = p_map.get("Bulk Jet Speed beta") {
                    probe.bulk_beta = b;
                }
                if let Some(&th) = p_map.get("Viewing Angle (deg)") {
                    probe.viewing_angle_deg = th;
                }
                if let Some(&p) = p_map.get("Electron Index p") {
                    probe.electron_index_p = p;
                }
                if let Some(&g) = p_map.get("Adiabatic Index") {
                    probe.adiabatic_index = g;
                }
                evaluate_jet_knot_observables(&Self::knot_config_from(&probe))
                    .map(|o| o.li_ma_sigma)
                    .unwrap_or(0.0)
            });
        crate::sensitivity::render_sensitivity_panel_ui(
            ui,
            &self.sensitivity_engine,
            "relativistic_jet_beaming_lab",
        );

        let next_idx = self.cfg.snapshot_manager.len() + 1;
        let summary = format!(
            "beta={:.2}, theta={:.0} deg, p={:.1}",
            self.cfg.bulk_beta, self.cfg.viewing_angle_deg, self.cfg.electron_index_p
        );
        let (series_name, pts) = match self.cfg.plot_mode {
            JetPlotMode::PolarTwinJetLobe => {
                ("Approaching Jet Lobe", self.approaching_lobe.clone())
            }
            JetPlotMode::AccretionRingProfile => {
                ("Accretion Ring I_nu / I_0", self.ring_profile.clone())
            }
        };
        if self.cfg.snapshot_manager.render_ui(ui, move || {
            Some((
                format!("Beaming {next_idx}"),
                summary,
                vec![TraceSeries::new(series_name, pts)],
            ))
        }) {
            self.revision = self.revision.wrapping_add(1).max(1);
        }
    }

    pub(super) fn render_beaming_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        match self.cfg.plot_mode {
            JetPlotMode::PolarTwinJetLobe => self.render_polar_lobe_plot(ui, &palette),
            JetPlotMode::AccretionRingProfile => self.render_ring_profile_plot(ui, &palette),
        }
    }

    fn render_polar_lobe_plot(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        let los_rad = self.cfg.viewing_angle_deg.to_radians();
        let r_max = 4.2_f64;
        let los_line = vec![[0.0, 0.0], [r_max * los_rad.cos(), r_max * los_rad.sin()]];
        Plot::new("relativistic_twin_jet_polar_plot")
            .data_aspect(1.0)
            .view_aspect(1.5)
            .x_axis_label("Jet Axis log10(1 + D^(3+alpha)) cos(theta)")
            .y_axis_label("Transverse log10(1 + D^(3+alpha)) sin(theta)")
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(
                        "Approaching Jet Lobe",
                        PlotPoints::new(self.approaching_lobe.clone()),
                    )
                    .color(palette.accent)
                    .width(2.5_f32),
                );
                plot_ui.line(
                    Line::new(
                        "Receding Counter-Jet Lobe",
                        PlotPoints::new(self.receding_lobe.clone()),
                    )
                    .color(palette.axis_x)
                    .style(LineStyle::Dashed { length: 6.0 })
                    .width(2.0_f32),
                );
                plot_ui.line(
                    Line::new("Observer Line of Sight (theta)", PlotPoints::new(los_line))
                        .color(palette.info)
                        .style(LineStyle::Dotted { spacing: 5.0 }),
                );
                self.cfg.snapshot_manager.render_on_plot(plot_ui);
            });
    }

    fn render_ring_profile_plot(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        let iso_ref = vec![[-180.0, 1.0], [180.0, 1.0]];
        Plot::new("relativistic_accretion_ring_profile_plot")
            .view_aspect(1.75)
            .x_axis_label("Ring Azimuthal Angle phi (deg)")
            .y_axis_label("Beamed Brightness Factor D(phi)^(3+alpha)")
            .include_y(0.0)
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(
                        "Accretion Ring Beaming I_nu / I_0",
                        PlotPoints::new(self.ring_profile.clone()),
                    )
                    .color(palette.accent)
                    .width(2.5_f32),
                );
                plot_ui.line(
                    Line::new("Unbeamed Rest-Frame Reference", PlotPoints::new(iso_ref))
                        .color(palette.info)
                        .style(LineStyle::Dashed { length: 6.0 }),
                );
                self.cfg.snapshot_manager.render_on_plot(plot_ui);
            });
    }
}

// [cite:general_relativity_geodesics]
