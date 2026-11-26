#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::clamp_finite;
use crate::framework::{ExportCapabilities, FiniteStateGuard, InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::physics::high_energy::{
    effective_potential, newtonian_periapsis_radius, GeodesicKind, SchwarzschildGeodesicBundle,
    MAX_GEODESIC_TRAJECTORIES,
};

pub(super) const GEODESIC_TRAIL_CAPACITY: usize = 900;
const POTENTIAL_SAMPLES: usize = 120;

const fn is_false(v: &bool) -> bool {
    !*v
}

/// Simulation scenario for [`BlackHoleGeodesicWorkbench`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum GeodesicWorkbenchMode {
    /// Parallel photon beam lensed past the photon sphere $r_{\text{ph}} = 1.5 R_s$.
    #[default]
    PhotonRingLensing,
    /// Massive test-particle orbit/plunge near $r_{\text{ISCO}} = 3 R_s$ vs. Newtonian $-GM/r^2$.
    IscoPlungeComparison,
    /// Simultaneous 64-trajectory mixed null and timelike relativistic ensemble.
    MixedEnsemble64,
}

impl GeodesicWorkbenchMode {
    /// Human-readable mode title for UI selectors.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::PhotonRingLensing => "Photon-Ring Lensing Fan",
            Self::IscoPlungeComparison => "Sub-ISCO Plunge vs. Newtonian",
            Self::MixedEnsemble64 => "64-Ray Null & Timelike Ensemble",
        }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BlackHoleGeodesicState {
    pub(crate) probe_b_over_rs: f64,
    pub(crate) launch_x_over_rs: f64,
    pub(crate) fan_half_span_over_rs: f64,
    pub(crate) ray_count: usize,
    pub(crate) r0_over_rs: f64,
    pub(crate) l_over_l_isco: f64,
    pub(crate) inward_nudge: f64,
    pub(crate) d_lambda: f64,
    pub(crate) substeps_per_frame: usize,
    pub(crate) mode: GeodesicWorkbenchMode,
    pub(crate) show_newtonian_overlay: bool,
    pub(crate) show_effective_potential: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub(crate) is_playing: bool,
}

/// Interactive Schwarzschild photon-ring lensing and ISCO plunge workbench.
pub struct BlackHoleGeodesicWorkbench {
    pub(crate) cfg: BlackHoleGeodesicState,
    pub(crate) bundle: SchwarzschildGeodesicBundle,
    pub(super) potential_gr: Vec<[f64; 2]>,
    pub(super) potential_newt: Vec<[f64; 2]>,
    pub(super) revision: u64,
}

impl Default for BlackHoleGeodesicWorkbench {
    fn default() -> Self {
        let cfg = BlackHoleGeodesicState {
            probe_b_over_rs: 2.60,
            launch_x_over_rs: 17.45,
            fan_half_span_over_rs: 0.80,
            ray_count: 9,
            r0_over_rs: 3.0,
            l_over_l_isco: 0.99,
            inward_nudge: 0.001,
            d_lambda: 0.05,
            substeps_per_frame: 8,
            mode: GeodesicWorkbenchMode::PhotonRingLensing,
            show_newtonian_overlay: true,
            show_effective_potential: false,
            is_playing: false,
        };
        let bundle = SchwarzschildGeodesicBundle::new(
            1.0,
            MAX_GEODESIC_TRAJECTORIES,
            GEODESIC_TRAIL_CAPACITY,
        )
        .expect("positive Schwarzschild scale");
        let mut tool = Self {
            cfg,
            bundle,
            potential_gr: vec![[0.0, 0.0]; POTENTIAL_SAMPLES],
            potential_newt: vec![[0.0, 0.0]; POTENTIAL_SAMPLES],
            revision: 1,
        };
        tool.rebuild_bundle(800);
        tool
    }
}

impl BlackHoleGeodesicWorkbench {
    pub(crate) fn rebuild_bundle(&mut self, prewarm_steps: usize) {
        self.revision = self.revision.wrapping_add(1).max(1);
        let c = &self.cfg;
        match c.mode {
            GeodesicWorkbenchMode::PhotonRingLensing => {
                self.bundle.seed_photon_lensing_fan(
                    c.ray_count,
                    c.probe_b_over_rs,
                    c.launch_x_over_rs,
                    c.fan_half_span_over_rs,
                );
            }
            GeodesicWorkbenchMode::IscoPlungeComparison => {
                self.bundle.seed_timelike_isco_comparison(
                    c.r0_over_rs,
                    c.l_over_l_isco,
                    c.inward_nudge,
                );
            }
            GeodesicWorkbenchMode::MixedEnsemble64 => {
                self.bundle.seed_mixed_ensemble(c.ray_count.max(2));
            }
        }
        if prewarm_steps > 0 {
            self.bundle.step_n(c.d_lambda, prewarm_steps);
        }
        self.refresh_potential_curves();
    }

    pub(super) fn refresh_potential_curves(&mut self) {
        let (kind, l_val) = match self.cfg.mode {
            GeodesicWorkbenchMode::PhotonRingLensing => {
                (GeodesicKind::Null, self.cfg.probe_b_over_rs)
            }
            GeodesicWorkbenchMode::IscoPlungeComparison
            | GeodesicWorkbenchMode::MixedEnsemble64 => (
                GeodesicKind::Timelike,
                self.cfg.l_over_l_isco * 3.0_f64.sqrt(),
            ),
        };
        for (k, (gr_pt, nw_pt)) in self
            .potential_gr
            .iter_mut()
            .zip(self.potential_newt.iter_mut())
            .enumerate()
        {
            let r = 1.05 + 10.95 * (k as f64) / ((POTENTIAL_SAMPLES - 1) as f64);
            let v_gr = effective_potential(1.0, r, l_val, kind).unwrap_or(0.0);
            let v_nw = if kind == GeodesicKind::Timelike {
                0.5 - 0.5 / r + 0.5 * (l_val * l_val) / (r * r)
            } else {
                0.5 * (l_val * l_val) / (r * r)
            };
            *gr_pt = [r, v_gr];
            *nw_pt = [r, v_nw];
        }
    }

    /// Executes a verified zero-allocation RK4 burst inside `VerifiedEngine::run_verified`.
    #[cfg(any(test, not(target_arch = "wasm32")))]
    pub(crate) fn step_inside_verified_engine(&mut self) -> (u64, u64) {
        self.bundle
            .step_n_verified(self.cfg.d_lambda, self.cfg.substeps_per_frame)
    }

    pub(crate) fn advance_burst(&mut self) {
        #[cfg(all(not(test), not(target_arch = "wasm32")))]
        {
            let _ = self.step_inside_verified_engine();
        }
        #[cfg(any(test, target_arch = "wasm32"))]
        {
            self.bundle
                .step_n(self.cfg.d_lambda, self.cfg.substeps_per_frame);
        }
    }

    pub(crate) fn analytical_newtonian_periapsis(&self) -> f64 {
        newtonian_periapsis_radius(self.cfg.r0_over_rs, self.cfg.l_over_l_isco)
    }

    pub(super) fn circle_ring_points(radius: f64, segments: usize) -> Vec<[f64; 2]> {
        (0..=segments)
            .map(|k| {
                let (s, c) = (std::f64::consts::TAU * (k as f64 / segments as f64)).sin_cos();
                [radius * c, radius * s]
            })
            .collect()
    }

    fn collect_geodesic_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        if self.cfg.show_effective_potential {
            return vec![
                (
                    "GR Effective Potential V_eff(r)".into(),
                    self.potential_gr.clone(),
                ),
                (
                    "Newtonian Potential Reference".into(),
                    self.potential_newt.clone(),
                ),
            ];
        }
        let mut out = vec![
            (
                "Event Horizon (R_s)".into(),
                Self::circle_ring_points(1.0, 48),
            ),
            (
                "Photon Sphere (1.5 R_s)".into(),
                Self::circle_ring_points(1.5, 48),
            ),
            ("ISCO (3.0 R_s)".into(), Self::circle_ring_points(3.0, 48)),
        ];
        let mut buf = vec![[0.0_f64; 2]; GEODESIC_TRAIL_CAPACITY];
        let n0 = self.bundle.copy_trajectory_trail(0, &mut buf);
        if n0 >= 2 {
            out.push(("Primary Relativistic Geodesic".into(), buf[..n0].to_vec()));
        }
        if self.cfg.show_newtonian_overlay {
            let nn = self.bundle.copy_newtonian_trail(&mut buf);
            if nn >= 2 {
                out.push(("Newtonian Trajectory (-GM/r^2)".into(), buf[..nn].to_vec()));
            }
        }
        out
    }
}

impl FiniteStateGuard for BlackHoleGeodesicWorkbench {
    fn is_finite_state(&self) -> bool {
        let c = &self.cfg;
        [
            c.probe_b_over_rs,
            c.launch_x_over_rs,
            c.fan_half_span_over_rs,
            c.r0_over_rs,
            c.l_over_l_isco,
            c.inward_nudge,
            c.d_lambda,
        ]
        .into_iter()
        .all(f64::is_finite)
            && c.d_lambda > 0.0
            && self
                .bundle
                .trajectories()
                .iter()
                .all(|t| t.pos.iter().all(|v| v.is_finite()))
    }

    fn auto_scale_step_size(&mut self) {
        self.cfg.d_lambda = (self.cfg.d_lambda * 0.5).clamp(0.005, 0.25);
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    fn clamp_parameters(&mut self) {
        let c = &mut self.cfg;
        c.probe_b_over_rs = clamp_finite(c.probe_b_over_rs, 0.3, 12.0, 2.60);
        c.launch_x_over_rs = clamp_finite(c.launch_x_over_rs, 5.0, 40.0, 17.45);
        c.fan_half_span_over_rs = clamp_finite(c.fan_half_span_over_rs, 0.0, 6.0, 0.80);
        c.ray_count = c.ray_count.clamp(1, MAX_GEODESIC_TRAJECTORIES);
        c.r0_over_rs = clamp_finite(c.r0_over_rs, 1.6, 20.0, 3.0);
        c.l_over_l_isco = clamp_finite(c.l_over_l_isco, 0.2, 2.0, 0.99);
        c.inward_nudge = clamp_finite(c.inward_nudge, 0.0, 0.4, 0.001);
        c.d_lambda = clamp_finite(c.d_lambda, 0.005, 0.25, 0.05);
        c.substeps_per_frame = c.substeps_per_frame.clamp(1, 32);
        self.rebuild_bundle(0);
    }
}

impl InteractiveTool for BlackHoleGeodesicWorkbench {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Black Hole Geodesic Workbench"
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
    fn step_simulation(&mut self) {
        self.advance_burst();
    }
    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.cfg).ok()
    }
    fn try_load_recovery_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        self.try_load_state(state)
    }
    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let s: BlackHoleGeodesicState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid Schwarzschild geodesic state: {e}"))?;
        let valid_floats = [
            (s.probe_b_over_rs, 0.3, 12.0),
            (s.launch_x_over_rs, 5.0, 40.0),
            (s.fan_half_span_over_rs, 0.0, 6.0),
            (s.r0_over_rs, 1.6, 20.0),
            (s.l_over_l_isco, 0.2, 2.0),
            (s.inward_nudge, 0.0, 0.4),
            (s.d_lambda, 0.005, 0.25),
        ]
        .into_iter()
        .all(|(v, lo, hi)| v.is_finite() && (lo..=hi).contains(&v));
        if !valid_floats
            || !(1..=MAX_GEODESIC_TRAJECTORIES).contains(&s.ray_count)
            || !(1..=32).contains(&s.substeps_per_frame)
        {
            return Err("Schwarzschild geodesic parameters out of valid physical bounds".into());
        }
        self.cfg = s;
        self.rebuild_bundle(800);
        Ok(())
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let datasets = egui_plot::get_last_plot_datasets();
        if datasets.is_empty() {
            self.collect_geodesic_datasets()
        } else {
            datasets
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
        if self.cfg.is_playing {
            self.advance_burst();
            ui.ctx().request_repaint();
        }
        let panel_id = crate::framework::scoped_panel_id(ui, "bh_geodesic_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Spacetime & Killing Readouts");
                self.render_transport_and_metrics(ui);
                ui.separator();
                if self.render_controls(ui) {
                    let prewarm = if self.cfg.is_playing { 0 } else { 800 };
                    self.rebuild_bundle(prewarm);
                }
            });
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Schwarzschild Geodesic & Photon-Ring Workbench");
            ui.label("Closed-form 4D Lorentzian connection tracing strong-field photon-sphere lensing and sub-ISCO relativistic plunges.");
            self.render_main_plot(ui);
        });
    }
}

// [cite:general_relativity_geodesics]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BlackHoleGeodesicWorkbench",
        display_name: "Black Hole Geodesic Workbench",
        domain: "astrophysics",
        tags: &["astrophysics", "black-hole", "general-relativity", "geodesics", "lensing"],
        build: || Box::new(BlackHoleGeodesicWorkbench::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for BlackHoleGeodesicWorkbench {
    fn theory_description(&self) -> String {
        format!(
            "Schwarzschild affine geodesic integration $d^2 x^\\mu / d\\lambda^2 + \\Gamma^\\mu_{{\\alpha\\beta}} u^\\alpha u^\\beta = 0$ in {} mode ($b / R_s = {:.3}$, $\\ell / \\ell_{{\\text{{ISCO}}}} = {:.3}$) contrasting photon-sphere capture ($b_{{\\text{{crit}}}} = 2.598\\,R_s$) and ISCO plunges ($r_{{\\text{{ISCO}}}} = 3\\,R_s$) against Newtonian gravity.",
            self.cfg.mode.label(),
            self.cfg.probe_b_over_rs,
            self.cfg.l_over_l_isco
        )
    }
    fn phonetic_description(&self) -> String {
        "Schwarzschild black hole photon ring lensing and ISCO geodesic workbench".into()
    }
    fn theory_citation(&self) -> String {
        "Schwarzschild (1916); Misner, Thorne & Wheeler (1973), Gravitation; Chandrasekhar (1983), The Mathematical Theory of Black Holes".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const PRESET_PHOTON_RING_LENSING: &str = concat!(
    r#"{"metadata":{"schema_version":"1.0","title":"Schwarzschild Photon Ring Lensing","domain":"astrophysics","timestamp":"2026-11-23T00:00:00Z"},"#,
    r#""config":{"probe_b_over_rs":2.6,"launch_x_over_rs":17.45,"fan_half_span_over_rs":0.8,"ray_count":9,"r0_over_rs":3.0,"l_over_l_isco":0.99,"#,
    r#""inward_nudge":0.001,"d_lambda":0.05,"substeps_per_frame":8,"mode":"PhotonRingLensing","show_newtonian_overlay":true,"show_effective_potential":false}}"#
);

const PRESET_ISCO_PLUNGE_CLIFF: &str = concat!(
    r#"{"metadata":{"schema_version":"1.0","title":"Sub-ISCO Plunge vs. Newtonian Orbit","domain":"astrophysics","timestamp":"2026-11-23T00:00:00Z"},"#,
    r#""config":{"probe_b_over_rs":2.6,"launch_x_over_rs":17.45,"fan_half_span_over_rs":0.8,"ray_count":9,"r0_over_rs":3.0,"l_over_l_isco":0.99,"#,
    r#""inward_nudge":0.001,"d_lambda":0.05,"substeps_per_frame":8,"mode":"IscoPlungeComparison","show_newtonian_overlay":true,"show_effective_potential":false}}"#
);

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "astrophysics_photon_ring_lensing_starter",
        title: "Schwarzschild Photon Ring Lensing",
        description: "Launch null rays near b_crit = 2.598 R_s to observe 6.51 rad relativistic loop deflection and horizon capture.",
        category: "Relativistic Astrophysics",
        icon: "🕳️",
        domain: "astrophysics",
        tool_name: Some("BlackHoleGeodesicWorkbench"),
        preset_generator: || PRESET_PHOTON_RING_LENSING,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "astrophysics_isco_plunge_cliff_starter",
        title: "Sub-ISCO Plunge vs. Newtonian Orbit",
        description: "Drop angular momentum 1% below L_ISCO at 3 R_s to contrast a 37.03 R_s/c horizon plunge with a bounded 2.88 R_s Newtonian ellipse.",
        category: "Relativistic Astrophysics",
        icon: "🌀",
        domain: "astrophysics",
        tool_name: Some("BlackHoleGeodesicWorkbench"),
        preset_generator: || PRESET_ISCO_PLUNGE_CLIFF,
    }
}
