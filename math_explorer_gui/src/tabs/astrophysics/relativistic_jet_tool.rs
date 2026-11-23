#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::clamp_finite;
use crate::framework::{
    ExportCapabilities, FiniteStateGuard, InteractiveTool, ToolStateBounds, TraceSnapshotManager,
};
use eframe::egui;
use math_explorer::physics::high_energy::{
    accretion_ring_beaming_factor, beamed_synchrotron_flux_factor, evaluate_jet_knot_observables,
    relativistic_doppler_factor, JetKnotConfig, JetKnotObservables,
};
use std::f64::consts::{PI, TAU};

const LOBE_SAMPLES: usize = 121;
const RING_AZIMUTH_SAMPLES: usize = 121;

/// Visualization mode for [`RelativisticJetBeamingLab`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum JetPlotMode {
    /// Polar log-intensity twin-jet Doppler beaming lobe $\log_{10}(1 + D^{3+\alpha})$.
    #[default]
    PolarTwinJetLobe,
    /// Azimuthal relativistic ISCO / accretion-ring brightness crescent $I_\nu(\varphi) / I_0$.
    AccretionRingProfile,
}

impl JetPlotMode {
    /// Human-readable mode label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::PolarTwinJetLobe => "Twin-Jet Polar Beaming Lobe",
            Self::AccretionRingProfile => "EHT Accretion-Ring Crescent Profile",
        }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RelativisticJetState {
    pub(crate) bulk_beta: f64,
    pub(crate) viewing_angle_deg: f64,
    pub(crate) disk_r_over_rs: f64,
    pub(crate) disk_inclination_deg: f64,
    pub(crate) electron_index_p: f64,
    pub(crate) electron_log10_gamma: f64,
    pub(crate) upstream_log10_ub: f64,
    pub(crate) adiabatic_index: f64,
    pub(crate) intrinsic_signal_counts: f64,
    pub(crate) background_counts_off: f64,
    pub(crate) exposure_ratio: f64,
    pub(crate) plot_mode: JetPlotMode,
    #[serde(default, skip_serializing_if = "TraceSnapshotManager::is_empty")]
    pub(crate) snapshot_manager: TraceSnapshotManager,
}

/// Interactive relativistic twin-jet Doppler beaming, shock compression, and Li & Ma gamma-ray lab.
pub struct RelativisticJetBeamingLab {
    pub(crate) cfg: RelativisticJetState,
    pub(crate) observables: JetKnotObservables,
    pub(super) approaching_lobe: Vec<[f64; 2]>,
    pub(super) receding_lobe: Vec<[f64; 2]>,
    pub(super) ring_profile: Vec<[f64; 2]>,
    pub(super) sensitivity_engine: crate::sensitivity::SensitivityEngine,
    pub(super) revision: u64,
}

impl Default for RelativisticJetBeamingLab {
    fn default() -> Self {
        let cfg = RelativisticJetState {
            bulk_beta: 0.90,
            viewing_angle_deg: 15.0,
            disk_r_over_rs: 3.0,
            disk_inclination_deg: 90.0,
            electron_index_p: 2.4,
            electron_log10_gamma: 3.0,
            upstream_log10_ub: -3.0,
            adiabatic_index: 4.0 / 3.0,
            intrinsic_signal_counts: 1.0,
            background_counts_off: 40.0,
            exposure_ratio: 1.0,
            plot_mode: JetPlotMode::PolarTwinJetLobe,
            snapshot_manager: TraceSnapshotManager::new(),
        };
        let obs = evaluate_jet_knot_observables(&Self::knot_config_from(&cfg))
            .expect("valid default jet knot config");
        let mut lab = Self {
            cfg,
            observables: obs,
            approaching_lobe: vec![[0.0, 0.0]; LOBE_SAMPLES],
            receding_lobe: vec![[0.0, 0.0]; LOBE_SAMPLES],
            ring_profile: vec![[0.0, 0.0]; RING_AZIMUTH_SAMPLES],
            sensitivity_engine: crate::sensitivity::SensitivityEngine::new(),
            revision: 1,
        };
        lab.recalculate();
        lab
    }
}

impl RelativisticJetBeamingLab {
    pub(crate) fn knot_config_from(c: &RelativisticJetState) -> JetKnotConfig {
        JetKnotConfig {
            bulk_beta: c.bulk_beta.clamp(0.05, 0.995),
            viewing_angle_rad: c.viewing_angle_deg.clamp(1.0, 89.0).to_radians(),
            disk_r_over_rs: c.disk_r_over_rs.clamp(1.55, 20.0),
            disk_inclination_rad: c.disk_inclination_deg.clamp(0.0, 90.0).to_radians(),
            electron_index_p: c.electron_index_p.clamp(1.2, 4.5),
            electron_gamma: 10.0_f64.powf(c.electron_log10_gamma.clamp(1.0, 6.0)),
            upstream_u_b: 10.0_f64.powf(c.upstream_log10_ub.clamp(-6.0, 2.0)),
            adiabatic_index: c.adiabatic_index.clamp(1.1, 2.0),
            intrinsic_signal_counts: c.intrinsic_signal_counts.clamp(0.01, 100.0),
            background_counts_off: c.background_counts_off.clamp(1.0, 5_000.0),
            exposure_ratio: c.exposure_ratio.clamp(0.1, 10.0),
        }
    }

    pub(crate) fn refresh_buffers(&mut self) {
        let kcfg = Self::knot_config_from(&self.cfg);
        if let Ok(obs) = evaluate_jet_knot_observables(&kcfg) {
            self.observables = obs;
        }
        let alpha = self.observables.spectral_index_alpha;
        for (i, (app_pt, rec_pt)) in self
            .approaching_lobe
            .iter_mut()
            .zip(self.receding_lobe.iter_mut())
            .enumerate()
        {
            let theta = TAU * (i as f64) / ((LOBE_SAMPLES - 1) as f64);
            let d_app = relativistic_doppler_factor(kcfg.bulk_beta, theta).unwrap_or(1.0);
            let d_rec = relativistic_doppler_factor(kcfg.bulk_beta, PI - theta).unwrap_or(1.0);
            let r_app =
                (1.0 + beamed_synchrotron_flux_factor(d_app, alpha, false).unwrap_or(1.0)).log10();
            let r_rec =
                (1.0 + beamed_synchrotron_flux_factor(d_rec, alpha, false).unwrap_or(1.0)).log10();
            let (sin_t, cos_t) = theta.sin_cos();
            *app_pt = [r_app * cos_t, r_app * sin_t];
            *rec_pt = [r_rec * cos_t, r_rec * sin_t];
        }
        for (i, ring_pt) in self.ring_profile.iter_mut().enumerate() {
            let phi_deg = -180.0 + 360.0 * (i as f64) / ((RING_AZIMUTH_SAMPLES - 1) as f64);
            let boost = accretion_ring_beaming_factor(
                kcfg.disk_r_over_rs,
                kcfg.disk_inclination_rad,
                phi_deg.to_radians(),
                kcfg.electron_index_p,
            )
            .unwrap_or(1.0);
            *ring_pt = [phi_deg, boost];
        }
    }

    pub(crate) fn recalculate(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
        self.refresh_buffers();
    }

    /// Evaluates observables and pre-allocated angular profiles inside `VerifiedEngine::run_verified`.
    #[cfg(test)]
    pub(crate) fn recalculate_inside_verified_engine(&mut self) -> (u64, u64) {
        let ((), metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            self.refresh_buffers();
        });
        (metrics.function_calls, metrics.arithmetic_ops)
    }

    fn apply_loaded_state(
        &mut self,
        mut s: RelativisticJetState,
        keep_snaps: bool,
    ) -> Result<(), String> {
        let valid = [
            (s.bulk_beta, 0.05, 0.995),
            (s.viewing_angle_deg, 1.0, 89.0),
            (s.disk_r_over_rs, 1.55, 20.0),
            (s.disk_inclination_deg, 0.0, 90.0),
            (s.electron_index_p, 1.2, 4.5),
            (s.electron_log10_gamma, 1.0, 6.0),
            (s.upstream_log10_ub, -6.0, 2.0),
            (s.adiabatic_index, 1.1, 2.0),
            (s.intrinsic_signal_counts, 0.01, 100.0),
            (s.background_counts_off, 1.0, 5_000.0),
            (s.exposure_ratio, 0.1, 10.0),
        ]
        .into_iter()
        .all(|(v, lo, hi)| v.is_finite() && (lo..=hi).contains(&v));
        if !valid {
            return Err("Relativistic jet parameters out of valid physical bounds".into());
        }
        if keep_snaps {
            s.snapshot_manager = std::mem::take(&mut self.cfg.snapshot_manager);
        }
        self.cfg = s;
        self.recalculate();
        Ok(())
    }
}

impl FiniteStateGuard for RelativisticJetBeamingLab {
    fn is_finite_state(&self) -> bool {
        let c = &self.cfg;
        [
            c.bulk_beta,
            c.viewing_angle_deg,
            c.disk_r_over_rs,
            c.disk_inclination_deg,
            c.electron_index_p,
            c.electron_log10_gamma,
            c.upstream_log10_ub,
            c.adiabatic_index,
            c.intrinsic_signal_counts,
            c.background_counts_off,
            c.exposure_ratio,
            self.observables.jet_flux_asymmetry,
            self.observables.li_ma_sigma,
        ]
        .into_iter()
        .all(f64::is_finite)
    }

    fn clamp_parameters(&mut self) {
        let c = &mut self.cfg;
        c.bulk_beta = clamp_finite(c.bulk_beta, 0.05, 0.995, 0.90);
        c.viewing_angle_deg = clamp_finite(c.viewing_angle_deg, 1.0, 89.0, 15.0);
        c.disk_r_over_rs = clamp_finite(c.disk_r_over_rs, 1.55, 20.0, 3.0);
        c.disk_inclination_deg = clamp_finite(c.disk_inclination_deg, 0.0, 90.0, 90.0);
        c.electron_index_p = clamp_finite(c.electron_index_p, 1.2, 4.5, 2.4);
        c.electron_log10_gamma = clamp_finite(c.electron_log10_gamma, 1.0, 6.0, 3.0);
        c.upstream_log10_ub = clamp_finite(c.upstream_log10_ub, -6.0, 2.0, -3.0);
        c.adiabatic_index = clamp_finite(c.adiabatic_index, 1.1, 2.0, 4.0 / 3.0);
        c.intrinsic_signal_counts = clamp_finite(c.intrinsic_signal_counts, 0.01, 100.0, 1.0);
        c.background_counts_off = clamp_finite(c.background_counts_off, 1.0, 5_000.0, 40.0);
        c.exposure_ratio = clamp_finite(c.exposure_ratio, 0.1, 10.0, 1.0);
        self.recalculate();
    }
}

impl InteractiveTool for RelativisticJetBeamingLab {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Relativistic Jet Beaming Lab"
    }
    fn domain(&self) -> &'static str {
        "astrophysics"
    }
    fn state_revision(&self) -> u64 {
        self.revision
    }
    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::default()
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
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
        let parsed: RelativisticJetState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid jet beaming recovery state: {e}"))?;
        self.apply_loaded_state(parsed, true)
    }
    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: RelativisticJetState =
            serde_json::from_str(state).map_err(|e| format!("Invalid jet beaming state: {e}"))?;
        self.apply_loaded_state(parsed, false)
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let datasets = egui_plot::get_last_plot_datasets();
        if !datasets.is_empty() {
            return datasets;
        }
        match self.cfg.plot_mode {
            JetPlotMode::PolarTwinJetLobe => vec![
                ("Approaching Jet Lobe".into(), self.approaching_lobe.clone()),
                (
                    "Receding Counter-Jet Lobe".into(),
                    self.receding_lobe.clone(),
                ),
            ],
            JetPlotMode::AccretionRingProfile => vec![(
                "Accretion Ring Beaming I_nu / I_0".into(),
                self.ring_profile.clone(),
            )],
        }
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
        let panel_id = crate::framework::scoped_panel_id(ui, "relativistic_jet_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.render_observables_readouts(ui);
                ui.separator();
                if self.render_controls(ui) {
                    self.recalculate();
                }
                ui.separator();
                self.render_sensitivity_and_snapshots(ui);
            });
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Relativistic Jet & Accretion-Ring Doppler Beaming");
            ui.label("Coupled special-relativistic Doppler beaming D^(3+alpha), Rankine-Hugoniot shock compression, synchrotron emission, and Li & Ma significance.");
            self.render_beaming_plot(ui);
        });
    }
}

// [cite:general_relativity_geodesics]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "RelativisticJetBeamingLab",
        display_name: "Relativistic Jet Beaming Lab",
        domain: "astrophysics",
        tags: &["astrophysics", "relativistic-jets", "synchrotron", "doppler-beaming", "gamma-ray"],
        build: || Box::new(RelativisticJetBeamingLab::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for RelativisticJetBeamingLab {
    fn theory_description(&self) -> String {
        format!(
            "Relativistic jet and accretion-ring Doppler beaming $S_\\nu = D^{{3 + \\alpha}} S_{{\\nu,0}}$ ($\\beta = {:.3}$, $\\theta = {:.1}^\\circ$, $\\alpha = {:.2}$) coupled to Rankine-Hugoniot shock compression $r = (\\hat{{\\gamma}} + 1)/(\\hat{{\\gamma}} - 1)$, synchrotron radiation, and Li & Ma (1983) likelihood significance.",
            self.cfg.bulk_beta,
            self.cfg.viewing_angle_deg,
            self.observables.spectral_index_alpha
        )
    }
    fn phonetic_description(&self) -> String {
        "Relativistic twin jet Doppler beaming shock synchrotron and Li Ma significance laboratory"
            .into()
    }
    fn theory_citation(&self) -> String {
        "Blandford & Konigl (1979), Relativistic Jets as Compact Radio Sources; Rybicki & Lightman (1979); Li & Ma (1983), ApJ 272, 317".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const PRESET_TWIN_JET_BEAMING: &str = concat!(
    r#"{"metadata":{"schema_version":"1.0","title":"M87 / Blazar Twin-Jet Asymmetry","domain":"astrophysics","timestamp":"2026-11-23T00:00:00Z"},"#,
    r#""config":{"bulk_beta":0.9,"viewing_angle_deg":15.0,"disk_r_over_rs":3.0,"disk_inclination_deg":90.0,"electron_index_p":2.4,"#,
    r#""electron_log10_gamma":3.0,"upstream_log10_ub":-3.0,"adiabatic_index":1.3333333333333333,"intrinsic_signal_counts":1.0,"background_counts_off":40.0,"exposure_ratio":1.0,"plot_mode":"PolarTwinJetLobe"}}"#
);

const PRESET_EHT_ACCRETION_RING: &str = concat!(
    r#"{"metadata":{"schema_version":"1.0","title":"EHT Edge-On ISCO Ring Beaming","domain":"astrophysics","timestamp":"2026-11-23T00:00:00Z"},"#,
    r#""config":{"bulk_beta":0.9,"viewing_angle_deg":15.0,"disk_r_over_rs":3.0,"disk_inclination_deg":90.0,"electron_index_p":2.4,"#,
    r#""electron_log10_gamma":3.0,"upstream_log10_ub":-3.0,"adiabatic_index":1.3333333333333333,"intrinsic_signal_counts":1.0,"background_counts_off":40.0,"exposure_ratio":1.0,"plot_mode":"AccretionRingProfile"}}"#
);

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "astrophysics_twin_jet_beaming_starter",
        title: "M87 / Blazar Twin-Jet Asymmetry",
        description: "Reproduce 18,855x approaching-to-receding twin-jet flux asymmetry at beta = 0.9, 15 deg viewing angle, and p = 2.4.",
        category: "Relativistic Astrophysics",
        icon: "⚡",
        domain: "astrophysics",
        tool_name: Some("RelativisticJetBeamingLab"),
        preset_generator: || PRESET_TWIN_JET_BEAMING,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "astrophysics_eht_accretion_ring_starter",
        title: "EHT Edge-On ISCO Ring Beaming",
        description: "Inspect the 58.3x Doppler crescent brightness asymmetry of an edge-on Schwarzschild ISCO ring (r = 3 R_s, beta_shell = 0.5).",
        category: "Relativistic Astrophysics",
        icon: "🔆",
        domain: "astrophysics",
        tool_name: Some("RelativisticJetBeamingLab"),
        preset_generator: || PRESET_EHT_ACCRETION_RING,
    }
}
