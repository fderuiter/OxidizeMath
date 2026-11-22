#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{HaloKindChoice, clamp_finite};
use crate::accessibility::AccessibleHoverText;
use crate::framework::{FiniteStateGuard, InteractiveTool, TraceSeries, TraceSnapshotManager};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, LineStyle, Plot, PlotPoints};
use math_explorer::physics::astrophysics::{
    BaryonicDiskConfig, DwarfMorphology, Magnitude, Mpc, VelocityDecomposition,
    decompose_rotation_curve, sample_rotation_curve_into,
};

const CURVE_SAMPLES: usize = 96;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RotationCurveState {
    pub(crate) distance_mpc: f64,
    pub(crate) abs_magnitude_v: f64,
    pub(crate) use_absolute_magnitude: bool,
    pub(crate) morphology: DwarfMorphology,
    pub(crate) gas_fraction: f64,
    pub(crate) disk_scale_kpc: f64,
    pub(crate) halo_kind: HaloKindChoice,
    pub(crate) halo_log10_mass: f64,
    pub(crate) halo_scale_kpc: f64,
    pub(crate) halo_concentration: f64,
    pub(crate) max_radius_kpc: f64,
    pub(crate) compare_counterpart: bool,
    #[serde(default, skip_serializing_if = "TraceSnapshotManager::is_empty")]
    pub(crate) snapshot_manager: TraceSnapshotManager,
}

/// Interactive Tillaboev (2025) dwarf-galaxy scaling & core-versus-cusp rotation curve explorer.
pub struct RotationCurveExplorer {
    pub(crate) cfg: RotationCurveState,
    samples: Vec<VelocityDecomposition>,
    counterpart_samples: Vec<VelocityDecomposition>,
    sensitivity_engine: crate::sensitivity::SensitivityEngine,
    revision: u64,
}

impl Default for RotationCurveExplorer {
    fn default() -> Self {
        let mut tool = Self {
            cfg: RotationCurveState {
                distance_mpc: 12.0,
                abs_magnitude_v: -15.5,
                use_absolute_magnitude: false,
                morphology: DwarfMorphology::TypeCode10,
                gas_fraction: 0.60,
                disk_scale_kpc: 0.80,
                halo_kind: HaloKindChoice::BurkertCore,
                halo_log10_mass: 10.0,
                halo_scale_kpc: 2.2,
                halo_concentration: 12.0,
                max_radius_kpc: 12.0,
                compare_counterpart: true,
                snapshot_manager: TraceSnapshotManager::new(),
            },
            samples: vec![VelocityDecomposition::default(); CURVE_SAMPLES],
            counterpart_samples: vec![VelocityDecomposition::default(); CURVE_SAMPLES],
            sensitivity_engine: crate::sensitivity::SensitivityEngine::new(),
            revision: 1,
        };
        tool.recalculate();
        tool
    }
}

impl RotationCurveExplorer {
    pub(crate) fn current_baryons(&self) -> BaryonicDiskConfig {
        let c = &self.cfg;
        if c.use_absolute_magnitude {
            c.morphology.baryonic_config_from_magnitude(Magnitude(c.abs_magnitude_v), c.gas_fraction, c.disk_scale_kpc)
        } else {
            c.morphology.baryonic_config_from_distance(Mpc(c.distance_mpc), c.gas_fraction, c.disk_scale_kpc)
        }
    }

    const fn counterpart_halo_choice(&self) -> HaloKindChoice {
        match self.cfg.halo_kind {
            HaloKindChoice::BurkertCore => HaloKindChoice::NfwCusp,
            HaloKindChoice::NfwCusp | HaloKindChoice::BaryonicOnly => HaloKindChoice::BurkertCore,
        }
    }

    pub(crate) fn recalculate(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
        self.samples.resize(CURVE_SAMPLES, VelocityDecomposition::default());
        self.counterpart_samples.resize(CURVE_SAMPLES, VelocityDecomposition::default());
        let baryons = self.current_baryons();
        let (m, rs, c) = (self.cfg.halo_log10_mass, self.cfg.halo_scale_kpc, self.cfg.halo_concentration);
        let r_max = self.cfg.max_radius_kpc.clamp(2.0, 25.0);
        let halo = self.cfg.halo_kind.build_profile(m, rs, c);
        sample_rotation_curve_into(0.05, r_max, &baryons, &halo, &mut self.samples);
        let alt = self.counterpart_halo_choice().build_profile(m, rs, c);
        sample_rotation_curve_into(0.05, r_max, &baryons, &alt, &mut self.counterpart_samples);
    }

    fn apply_loaded_state(&mut self, mut s: RotationCurveState, keep_snaps: bool) -> Result<(), String> {
        let valid = [
            (s.distance_mpc, 0.5, 250.0),
            (s.abs_magnitude_v, -22.0, -8.0),
            (s.gas_fraction, 0.0, 0.95),
            (s.disk_scale_kpc, 0.2, 5.0),
            (s.halo_log10_mass, 8.0, 12.0),
            (s.halo_scale_kpc, 0.3, 10.0),
            (s.halo_concentration, 4.0, 25.0),
            (s.max_radius_kpc, 2.0, 25.0),
        ]
        .into_iter()
        .all(|(v, lo, hi)| v.is_finite() && (lo..=hi).contains(&v));
        if !valid {
            return Err("Rotation curve parameters out of valid physical bounds".into());
        }
        if keep_snaps {
            s.snapshot_manager = std::mem::take(&mut self.cfg.snapshot_manager);
        }
        self.cfg = s;
        self.recalculate();
        Ok(())
    }

    fn render_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let c = &mut self.cfg;
        let mut ch = false;
        ui.heading("Baryonic Scaling");
        egui::ComboBox::from_label("Dwarf Morphology")
            .selected_text(c.morphology.label())
            .show_ui(ui, |ui| {
                for m in [DwarfMorphology::GeneralIrregular, DwarfMorphology::TypeCode10, DwarfMorphology::TypeCode95To99] {
                    ch |= ui.selectable_value(&mut c.morphology, m, m.label()).changed();
                }
            });
        ch |= ui.checkbox(&mut c.use_absolute_magnitude, "Drive Mass via V-Band Magnitude (M_V)").changed();
        if c.use_absolute_magnitude {
            ch |= ui.add(egui::Slider::new(&mut c.abs_magnitude_v, -22.0..=-8.0).text("Abs Magnitude M_V")).changed();
        } else {
            ch |= ui.add(egui::Slider::new(&mut c.distance_mpc, 0.5..=250.0).text("Distance D (Mpc)")).changed();
        }
        ch |= ui.add(egui::Slider::new(&mut c.gas_fraction, 0.0..=0.95).text("HI Gas Fraction")).changed();
        ch |= ui.add(egui::Slider::new(&mut c.disk_scale_kpc, 0.2..=5.0).text("Disk Scale R_d (kpc)")).changed();
        ui.separator();
        ui.heading("Dark-Matter Halo");
        egui::ComboBox::from_label("Halo Profile")
            .selected_text(c.halo_kind.label())
            .show_ui(ui, |ui| {
                for h in [HaloKindChoice::BaryonicOnly, HaloKindChoice::NfwCusp, HaloKindChoice::BurkertCore] {
                    ch |= ui.selectable_value(&mut c.halo_kind, h, h.label()).changed();
                }
            });
        ch |= ui.add(egui::Slider::new(&mut c.halo_log10_mass, 8.0..=12.0).text("Halo log10(M200 / M_sun)")).changed();
        ch |= ui.add(egui::Slider::new(&mut c.halo_scale_kpc, 0.3..=10.0).text("Scale / Core Radius (kpc)")).changed();
        ch |= ui.add(egui::Slider::new(&mut c.halo_concentration, 4.0..=25.0).text("Virial Concentration c")).changed();
        ch |= ui.add(egui::Slider::new(&mut c.max_radius_kpc, 2.0..=25.0).text("Max Plot Radius (kpc)")).changed();
        ch |= ui.checkbox(&mut c.compare_counterpart, "Overlay Counterpart Halo Curve").changed();
        ch
    }

    fn render_tillaboev_readouts(&self, ui: &mut egui::Ui, baryons: &BaryonicDiskConfig) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.heading("Tillaboev (2025) Regressions");
        render_copyable_metric(ui, "log10(M_bary / M_sun)", &format!("{:.4}", baryons.scaling.log_mass_solar));
        render_copyable_metric(ui, "Total Baryonic Mass", &format!("{:.3e} M_sun", baryons.scaling.total_baryonic_mass_solar));
        render_copyable_metric(ui, "Predicted Redshift (z)", &format!("{:.5}", baryons.scaling.redshift));

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

    fn render_sensitivity_and_snapshots(&mut self, ui: &mut egui::Ui) {
        let c = &self.cfg;
        let params = [
            ("Halo log10(M200 / M_sun)", c.halo_log10_mass),
            ("Scale / Core Radius (kpc)", c.halo_scale_kpc),
            ("Disk Scale R_d (kpc)", c.disk_scale_kpc),
            ("HI Gas Fraction", c.gas_fraction),
        ];
        let ctx_bits = crate::sensitivity::canonical_f64_bits(c.max_radius_kpc);
        let (morph, dist, hk, conc) = (c.morphology, c.distance_mpc, c.halo_kind, c.halo_concentration);
        let r_eval = c.max_radius_kpc.clamp(2.0, 25.0);
        self.sensitivity_engine.compute_with_context(&params, &ctx_bits, |p_map| {
            let f_gas = p_map.get("HI Gas Fraction").copied().unwrap_or(0.6);
            let r_d = p_map.get("Disk Scale R_d (kpc)").copied().unwrap_or(0.8);
            let log_m = p_map.get("Halo log10(M200 / M_sun)").copied().unwrap_or(10.0);
            let r_s = p_map.get("Scale / Core Radius (kpc)").copied().unwrap_or(2.2);
            let bary = morph.baryonic_config_from_distance(Mpc(dist), f_gas, r_d);
            decompose_rotation_curve(r_eval, &bary, &hk.build_profile(log_m, r_s, conc)).v_circ_kms
        });
        crate::sensitivity::render_sensitivity_panel_ui(ui, &self.sensitivity_engine, "rotation_curve_explorer");

        let summary = format!("{}, {}, r_s={:.1} kpc", self.cfg.morphology.label(), self.cfg.halo_kind.label(), self.cfg.halo_scale_kpc);
        let next_idx = self.cfg.snapshot_manager.len() + 1;
        let pts: Vec<[f64; 2]> = self.samples.iter().map(|s| [s.radius_kpc, s.v_circ_kms]).collect();
        if self.cfg.snapshot_manager.render_ui(ui, move || {
            Some((format!("Curve {next_idx}"), summary, vec![TraceSeries::new("Total v_circ (km/s)", pts)]))
        }) {
            self.revision = self.revision.wrapping_add(1).max(1);
        }
    }

    fn render_curve_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let total_pts: Vec<[f64; 2]> = self.samples.iter().map(|s| [s.radius_kpc, s.v_circ_kms]).collect();
        let disk_pts: Vec<[f64; 2]> = self.samples.iter().map(|s| [s.radius_kpc, s.v_disk_kms]).collect();
        let gas_pts: Vec<[f64; 2]> = self.samples.iter().map(|s| [s.radius_kpc, s.v_gas_kms]).collect();
        let halo_pts: Vec<[f64; 2]> = self.samples.iter().map(|s| [s.radius_kpc, s.v_halo_kms]).collect();
        let alt_label = format!("Counterpart: {}", self.counterpart_halo_choice().label());
        let alt_pts: Vec<[f64; 2]> = self.counterpart_samples.iter().map(|s| [s.radius_kpc, s.v_circ_kms]).collect();

        Plot::new("astrophysics_rotation_curve_plot")
            .view_aspect(1.85)
            .x_axis_label("Galactocentric Radius r (kpc)")
            .y_axis_label("Circular Velocity v (km/s)")
            .include_y(0.0)
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(Line::new("Total v_circ (km/s)", PlotPoints::new(total_pts)).color(palette.accent).width(2.5_f32));
                plot_ui.line(Line::new("Stellar Disk v_disk", PlotPoints::new(disk_pts)).color(palette.axis_x));
                plot_ui.line(Line::new("HI Gas v_gas", PlotPoints::new(gas_pts)).color(palette.axis_y));
                plot_ui.line(Line::new("Dark Halo v_halo", PlotPoints::new(halo_pts)).color(palette.axis_z));
                if self.cfg.compare_counterpart {
                    plot_ui.line(Line::new(alt_label, PlotPoints::new(alt_pts)).color(palette.info).style(LineStyle::Dashed { length: 8.0 }));
                }
                self.cfg.snapshot_manager.render_on_plot(plot_ui);
            });
    }
}

impl FiniteStateGuard for RotationCurveExplorer {
    fn is_finite_state(&self) -> bool {
        let c = &self.cfg;
        [c.distance_mpc, c.abs_magnitude_v, c.gas_fraction, c.disk_scale_kpc, c.halo_log10_mass, c.halo_scale_kpc, c.halo_concentration, c.max_radius_kpc]
            .into_iter()
            .all(f64::is_finite)
            && self.samples.iter().all(|s| s.v_circ_kms.is_finite())
    }

    fn clamp_parameters(&mut self) {
        let c = &mut self.cfg;
        c.distance_mpc = clamp_finite(c.distance_mpc, 0.5, 250.0, 12.0);
        c.abs_magnitude_v = clamp_finite(c.abs_magnitude_v, -22.0, -8.0, -15.5);
        c.gas_fraction = clamp_finite(c.gas_fraction, 0.0, 0.95, 0.60);
        c.disk_scale_kpc = clamp_finite(c.disk_scale_kpc, 0.2, 5.0, 0.80);
        c.halo_log10_mass = clamp_finite(c.halo_log10_mass, 8.0, 12.0, 10.0);
        c.halo_scale_kpc = clamp_finite(c.halo_scale_kpc, 0.3, 10.0, 2.2);
        c.halo_concentration = clamp_finite(c.halo_concentration, 4.0, 25.0, 12.0);
        c.max_radius_kpc = clamp_finite(c.max_radius_kpc, 2.0, 25.0, 12.0);
        self.recalculate();
    }
}

impl InteractiveTool for RotationCurveExplorer {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Rotation Curve Explorer"
    }
    fn domain(&self) -> &'static str {
        "astrophysics"
    }
    fn state_revision(&self) -> u64 {
        self.revision
    }
    fn has_finite_parameters(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }
    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }
    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.cfg).ok()
    }
    fn try_load_recovery_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: RotationCurveState = serde_json::from_str(state).map_err(|e| format!("Invalid rotation curve recovery state: {e}"))?;
        self.apply_loaded_state(parsed, true)
    }
    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: RotationCurveState = serde_json::from_str(state).map_err(|e| format!("Invalid rotation curve state: {e}"))?;
        self.apply_loaded_state(parsed, false)
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let datasets = egui_plot::get_last_plot_datasets();
        if !datasets.is_empty() {
            return datasets;
        }
        vec![
            ("Total v_circ (km/s)".into(), self.samples.iter().map(|s| [s.radius_kpc, s.v_circ_kms]).collect()),
            ("Stellar Disk v_disk".into(), self.samples.iter().map(|s| [s.radius_kpc, s.v_disk_kms]).collect()),
            ("HI Gas v_gas".into(), self.samples.iter().map(|s| [s.radius_kpc, s.v_gas_kms]).collect()),
            ("Dark Halo v_halo".into(), self.samples.iter().map(|s| [s.radius_kpc, s.v_halo_kms]).collect()),
        ]
    }
    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        if !self.has_finite_parameters() {
            FiniteStateGuard::clamp_parameters(self);
        }
        let panel_id = crate::framework::scoped_panel_id(ui, "rotation_curve_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                let baryons_before = self.current_baryons();
                self.render_tillaboev_readouts(ui, &baryons_before);
                ui.separator();
                if self.render_controls(ui) {
                    self.recalculate();
                }
                ui.separator();
                self.render_sensitivity_and_snapshots(ui);
            });
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Dwarf-Galaxy Rotation Curve Decomposition");
            ui.label("Compare Keplerian baryonic falloff against cored (Burkert) and cuspy (NFW) dark-matter halos.");
            self.render_curve_plot(ui);
        });
    }
}

// [cite:dwarf_galaxy_empirical_dependencies]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "RotationCurveExplorer",
        display_name: "Rotation Curve Explorer",
        domain: "astrophysics",
        tags: &["astrophysics", "dark-matter", "rotation-curves", "galaxies"],
        build: || Box::new(RotationCurveExplorer::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for RotationCurveExplorer {
    fn theory_description(&self) -> String {
        format!(
            "Irregular dwarf-galaxy circular velocity decomposition $v_{{\\text{{circ}}}}(r) = \\sqrt{{v_{{\\text{{disk}}}}^2(r) + v_{{\\text{{gas}}}}^2(r) + v_{{\\text{{halo}}}}^2(r)}}$ pairing empirical baryonic scaling laws ({}) with {} halo kinematics ($\\log_{{10}} M_{{200}} = {:.2}$, $r_s = {:.2}\\,\\text{{kpc}}$).",
            self.cfg.morphology.label(),
            self.cfg.halo_kind.label(),
            self.cfg.halo_log10_mass,
            self.cfg.halo_scale_kpc
        )
    }
    fn phonetic_description(&self) -> String {
        "Dwarf galaxy baryonic and dark matter halo rotation curve decomposition".into()
    }
    fn theory_citation(&self) -> String {
        "Tillaboev, Tadjibaev & Otojanova (2025), Empirical dependencies for irregular dwarf galaxies (tillaboev2025); Burkert (1995); Navarro, Frenk & White (1997)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const PRESET_CORE_VS_CUSP: &str = concat!(
    r#"{"metadata":{"schema_version":"1.0","title":"Dwarf Core vs. Cusp Curve","domain":"astrophysics","timestamp":"2026-11-21T00:00:00Z"},"#,
    r#""config":{"distance_mpc":18.0,"abs_magnitude_v":-16.0,"use_absolute_magnitude":false,"morphology":"TypeCode10","gas_fraction":0.65,"#,
    r#""disk_scale_kpc":0.75,"halo_kind":"BurkertCore","halo_log10_mass":10.2,"halo_scale_kpc":2.0,"halo_concentration":13.0,"max_radius_kpc":14.0,"compare_counterpart":true}}"#
);

const PRESET_KEPLERIAN_FALLOFF: &str = concat!(
    r#"{"metadata":{"schema_version":"1.0","title":"Keplerian Baryonic Falloff","domain":"astrophysics","timestamp":"2026-11-21T00:00:00Z"},"#,
    r#""config":{"distance_mpc":9.5,"abs_magnitude_v":-17.2,"use_absolute_magnitude":true,"morphology":"GeneralIrregular","gas_fraction":0.45,"#,
    r#""disk_scale_kpc":0.65,"halo_kind":"BaryonicOnly","halo_log10_mass":9.8,"halo_scale_kpc":2.5,"halo_concentration":10.0,"max_radius_kpc":16.0,"compare_counterpart":true}}"#
);

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "astrophysics_core_vs_cusp_starter",
        title: "Dwarf Core vs. Cusp Curve",
        description: "Compare cored Burkert (1995) and cuspy NFW dark-matter halos over Tillaboev (2025) Type 10 dwarf baryons.",
        category: "Galactic Dynamics",
        icon: "🌌",
        domain: "astrophysics",
        tool_name: Some("RotationCurveExplorer"),
        preset_generator: || PRESET_CORE_VS_CUSP,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "astrophysics_keplerian_falloff_starter",
        title: "Keplerian Baryonic Falloff",
        description: "Inspect pure stellar + HI gas r^(-1/2) Keplerian rotation falloff without a dark-matter halo.",
        category: "Galactic Dynamics",
        icon: "🪐",
        domain: "astrophysics",
        tool_name: Some("RotationCurveExplorer"),
        preset_generator: || PRESET_KEPLERIAN_FALLOFF,
    }
}
