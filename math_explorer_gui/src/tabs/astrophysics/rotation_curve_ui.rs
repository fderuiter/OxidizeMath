#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::rotation_curve_tool::RotationCurveExplorer;
use super::HaloKindChoice;
use crate::accessibility::AccessibleHoverText;
use crate::framework::TraceSeries;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, LineStyle, Plot, PlotPoints};
use math_explorer::physics::astrophysics::{
    decompose_rotation_curve, BaryonicDiskConfig, DwarfMorphology, Mpc,
};

impl RotationCurveExplorer {
    pub(super) fn render_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut ch = self.render_baryonic_controls(ui);
        let c = &mut self.cfg;
        ui.separator();
        ui.heading("Dark-Matter Halo");
        egui::ComboBox::from_label("Halo Profile")
            .selected_text(c.halo_kind.label())
            .show_ui(ui, |ui| {
                for h in [
                    HaloKindChoice::BaryonicOnly,
                    HaloKindChoice::NfwCusp,
                    HaloKindChoice::BurkertCore,
                ] {
                    ch |= ui
                        .selectable_value(&mut c.halo_kind, h, h.label())
                        .changed();
                }
            });
        ch |= ui
            .add(
                egui::Slider::new(&mut c.halo_log10_mass, 8.0..=12.0)
                    .text("Halo log10(M200 / M_sun)"),
            )
            .changed();
        ch |= ui
            .add(
                egui::Slider::new(&mut c.halo_scale_kpc, 0.3..=10.0)
                    .text("Scale / Core Radius (kpc)"),
            )
            .changed();
        ch |= ui
            .add(
                egui::Slider::new(&mut c.halo_concentration, 4.0..=25.0)
                    .text("Virial Concentration c"),
            )
            .changed();
        ch |= ui
            .add(egui::Slider::new(&mut c.max_radius_kpc, 2.0..=25.0).text("Max Plot Radius (kpc)"))
            .changed();
        ch |= ui
            .checkbox(&mut c.compare_counterpart, "Overlay Counterpart Halo Curve")
            .changed();
        ch
    }

    fn render_baryonic_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let c = &mut self.cfg;
        let mut ch = false;
        ui.heading("Baryonic Scaling");
        egui::ComboBox::from_label("Dwarf Morphology")
            .selected_text(c.morphology.label())
            .show_ui(ui, |ui| {
                for m in [
                    DwarfMorphology::GeneralIrregular,
                    DwarfMorphology::TypeCode10,
                    DwarfMorphology::TypeCode95To99,
                ] {
                    ch |= ui
                        .selectable_value(&mut c.morphology, m, m.label())
                        .changed();
                }
            });
        ch |= ui
            .checkbox(
                &mut c.use_absolute_magnitude,
                "Drive Mass via V-Band Magnitude (M_V)",
            )
            .changed();
        if c.use_absolute_magnitude {
            ch |= ui
                .add(
                    egui::Slider::new(&mut c.abs_magnitude_v, -22.0..=-8.0)
                        .text("Abs Magnitude M_V"),
                )
                .changed();
        } else {
            ch |= ui
                .add(egui::Slider::new(&mut c.distance_mpc, 0.5..=250.0).text("Distance D (Mpc)"))
                .changed();
        }
        ch |= ui
            .add(egui::Slider::new(&mut c.gas_fraction, 0.0..=0.95).text("HI Gas Fraction"))
            .changed();
        ch |= ui
            .add(egui::Slider::new(&mut c.disk_scale_kpc, 0.2..=5.0).text("Disk Scale R_d (kpc)"))
            .changed();
        ch
    }

    pub(super) fn render_tillaboev_readouts(
        &self,
        ui: &mut egui::Ui,
        baryons: &BaryonicDiskConfig,
    ) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.heading("Tillaboev (2025) Regressions");
        render_copyable_metric(
            ui,
            "log10(M_bary / M_sun)",
            &format!("{:.4}", baryons.scaling.log_mass_solar),
        );
        render_copyable_metric(
            ui,
            "Total Baryonic Mass",
            &format!("{:.3e} M_sun", baryons.scaling.total_baryonic_mass_solar),
        );
        render_copyable_metric(
            ui,
            "Predicted Redshift (z)",
            &format!("{:.5}", baryons.scaling.redshift),
        );

        if let Some(m_b) = baryons.scaling.apparent_magnitude_b {
            render_copyable_metric(ui, "Apparent Magnitude (m_B)", &format!("{:.3} mag", m_b));
        } else {
            let badge_text = if self.cfg.use_absolute_magnitude {
                "ℹ Apparent Magnitude (m_B): Select Distance D (Mpc) mode to evaluate m_B(D)"
            } else {
                "⚠ Apparent Magnitude (m_B): Unestablished in Tillaboev (2025) for Type Code 9.5–9.9"
            };
            egui::Frame::group(ui.style())
                .stroke(egui::Stroke::new(1.0_f32, palette.warning))
                .show(ui, |ui| {
                    let resp = ui.add(
                        egui::Label::new(egui::RichText::new(badge_text).color(palette.warning))
                            .sense(egui::Sense::focusable_noninteractive()),
                    );
                    let _ = resp.accessible_hover_text(
                        "Tillaboev et al. (2025) report no established m_B(D) linear regression for Type Code 9.5 to 9.9 irregular dwarf galaxies.",
                    );
                });
        }
    }

    pub(super) fn render_sensitivity_and_snapshots(&mut self, ui: &mut egui::Ui) {
        let c = &self.cfg;
        let params = [
            ("Halo log10(M200 / M_sun)", c.halo_log10_mass),
            ("Scale / Core Radius (kpc)", c.halo_scale_kpc),
            ("Disk Scale R_d (kpc)", c.disk_scale_kpc),
            ("HI Gas Fraction", c.gas_fraction),
        ];
        let ctx_bits = crate::sensitivity::canonical_f64_bits(c.max_radius_kpc);
        let (morph, dist, hk, conc) = (
            c.morphology,
            c.distance_mpc,
            c.halo_kind,
            c.halo_concentration,
        );
        let r_eval = c.max_radius_kpc.clamp(2.0, 25.0);
        self.sensitivity_engine
            .compute_with_context(&params, &ctx_bits, |p_map| {
                let f_gas = p_map.get("HI Gas Fraction").copied().unwrap_or(0.6);
                let r_d = p_map.get("Disk Scale R_d (kpc)").copied().unwrap_or(0.8);
                let log_m = p_map
                    .get("Halo log10(M200 / M_sun)")
                    .copied()
                    .unwrap_or(10.0);
                let r_s = p_map
                    .get("Scale / Core Radius (kpc)")
                    .copied()
                    .unwrap_or(2.2);
                let bary = morph.baryonic_config_from_distance(Mpc(dist), f_gas, r_d);
                decompose_rotation_curve(r_eval, &bary, &hk.build_profile(log_m, r_s, conc))
                    .v_circ_kms
            });
        crate::sensitivity::render_sensitivity_panel_ui(
            ui,
            &self.sensitivity_engine,
            "rotation_curve_explorer",
        );

        let summary = format!(
            "{}, {}, r_s={:.1} kpc",
            self.cfg.morphology.label(),
            self.cfg.halo_kind.label(),
            self.cfg.halo_scale_kpc
        );
        let next_idx = self.cfg.snapshot_manager.len() + 1;
        let pts: Vec<[f64; 2]> = self
            .samples
            .iter()
            .map(|s| [s.radius_kpc, s.v_circ_kms])
            .collect();
        if self.cfg.snapshot_manager.render_ui(ui, move || {
            Some((
                format!("Curve {next_idx}"),
                summary,
                vec![TraceSeries::new("Total v_circ (km/s)", pts)],
            ))
        }) {
            self.revision = self.revision.wrapping_add(1).max(1);
        }
    }

    pub(super) fn render_curve_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let total_pts: Vec<[f64; 2]> = self
            .samples
            .iter()
            .map(|s| [s.radius_kpc, s.v_circ_kms])
            .collect();
        let disk_pts: Vec<[f64; 2]> = self
            .samples
            .iter()
            .map(|s| [s.radius_kpc, s.v_disk_kms])
            .collect();
        let gas_pts: Vec<[f64; 2]> = self
            .samples
            .iter()
            .map(|s| [s.radius_kpc, s.v_gas_kms])
            .collect();
        let halo_pts: Vec<[f64; 2]> = self
            .samples
            .iter()
            .map(|s| [s.radius_kpc, s.v_halo_kms])
            .collect();
        let alt_label = format!("Counterpart: {}", self.counterpart_halo_choice().label());
        let alt_pts: Vec<[f64; 2]> = self
            .counterpart_samples
            .iter()
            .map(|s| [s.radius_kpc, s.v_circ_kms])
            .collect();

        Plot::new("astrophysics_rotation_curve_plot")
            .view_aspect(1.85)
            .x_axis_label("Galactocentric Radius r (kpc)")
            .y_axis_label("Circular Velocity v (km/s)")
            .include_y(0.0)
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Total v_circ (km/s)", PlotPoints::new(total_pts))
                        .color(palette.accent)
                        .width(2.5_f32),
                );
                plot_ui.line(
                    Line::new("Stellar Disk v_disk", PlotPoints::new(disk_pts))
                        .color(palette.axis_x),
                );
                plot_ui.line(
                    Line::new("HI Gas v_gas", PlotPoints::new(gas_pts)).color(palette.axis_y),
                );
                plot_ui.line(
                    Line::new("Dark Halo v_halo", PlotPoints::new(halo_pts)).color(palette.axis_z),
                );
                if self.cfg.compare_counterpart {
                    plot_ui.line(
                        Line::new(alt_label, PlotPoints::new(alt_pts))
                            .color(palette.info)
                            .style(LineStyle::Dashed { length: 8.0 }),
                    );
                }
                self.cfg.snapshot_manager.render_on_plot(plot_ui);
            });
    }
}

// [cite:dwarf_galaxy_empirical_dependencies]
