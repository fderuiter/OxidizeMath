#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{HaloKindChoice, clamp_finite};
use crate::framework::{FiniteStateGuard, InteractiveTool, PlaybackAction, PlaybackTransport};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, LineStyle, MarkerShape, Plot, PlotPoints, Points};
use math_explorer::physics::astrophysics::{
    DwarfMorphology, GalacticDiskSystem, Mpc, OrbitParticleKind, OrbitPopulationConfig,
    decompose_rotation_curve,
};

const TRAIL_CAPACITY: usize = 64;

const fn is_false(v: &bool) -> bool {
    !*v
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GalacticOrbitsState {
    pub(crate) distance_mpc: f64,
    pub(crate) morphology: DwarfMorphology,
    pub(crate) gas_fraction: f64,
    pub(crate) disk_scale_kpc: f64,
    pub(crate) halo_kind: HaloKindChoice,
    pub(crate) halo_log10_mass: f64,
    pub(crate) halo_scale_kpc: f64,
    pub(crate) halo_concentration: f64,
    pub(crate) dt_myr: f64,
    pub(crate) substeps_per_frame: usize,
    pub(crate) disk_tracers: usize,
    pub(crate) tidal_tracers: usize,
    pub(crate) tidal_tangential_ratio: f64,
    #[serde(default, skip_serializing_if = "is_false")]
    pub(crate) is_playing: bool,
}

/// Interactive symplectic leapfrog dwarf-galaxy disk and tidal-stream orbit simulator.
pub struct GalacticOrbitSimulator {
    pub(crate) cfg: GalacticOrbitsState,
    pub(crate) system: GalacticDiskSystem,
    revision: u64,
}

impl Default for GalacticOrbitSimulator {
    fn default() -> Self {
        let cfg = GalacticOrbitsState {
            distance_mpc: 12.0,
            morphology: DwarfMorphology::TypeCode10,
            gas_fraction: 0.60,
            disk_scale_kpc: 0.80,
            halo_kind: HaloKindChoice::BurkertCore,
            halo_log10_mass: 10.1,
            halo_scale_kpc: 2.4,
            halo_concentration: 12.0,
            dt_myr: 1.5,
            substeps_per_frame: 4,
            disk_tracers: 18,
            tidal_tracers: 8,
            tidal_tangential_ratio: 0.62,
            is_playing: false,
        };
        let halo = cfg.halo_kind.build_profile(cfg.halo_log10_mass, cfg.halo_scale_kpc, cfg.halo_concentration);
        let mut system = GalacticDiskSystem::from_morphology(
            cfg.morphology,
            Mpc(cfg.distance_mpc),
            cfg.gas_fraction,
            cfg.disk_scale_kpc,
            halo,
            OrbitPopulationConfig {
                disk_tracers: cfg.disk_tracers,
                tidal_tracers: cfg.tidal_tracers,
                trail_capacity: TRAIL_CAPACITY,
                tidal_tangential_ratio: cfg.tidal_tangential_ratio,
            },
        );
        system.step_n_leapfrog(cfg.dt_myr, 16);
        Self { cfg, system, revision: 1 }
    }
}

impl GalacticOrbitSimulator {
    pub(crate) fn rebuild_system(&mut self, prewarm_steps: usize) {
        self.revision = self.revision.wrapping_add(1).max(1);
        let c = &self.cfg;
        let halo = c.halo_kind.build_profile(c.halo_log10_mass, c.halo_scale_kpc, c.halo_concentration);
        self.system = GalacticDiskSystem::from_morphology(
            c.morphology,
            Mpc(c.distance_mpc),
            c.gas_fraction,
            c.disk_scale_kpc,
            halo,
            OrbitPopulationConfig {
                disk_tracers: c.disk_tracers,
                tidal_tracers: c.tidal_tracers,
                trail_capacity: TRAIL_CAPACITY,
                tidal_tangential_ratio: c.tidal_tangential_ratio,
            },
        );
        if prewarm_steps > 0 {
            self.system.step_n_leapfrog(self.cfg.dt_myr, prewarm_steps);
        }
    }

    /// Executes a verified zero-allocation leapfrog burst inside `VerifiedEngine::run_verified`.
    #[cfg(any(test, not(target_arch = "wasm32")))]
    pub(crate) fn step_inside_verified_engine(&mut self) -> (u64, u64) {
        self.system
            .step_n_verified(self.cfg.dt_myr, self.cfg.substeps_per_frame)
    }

    pub(crate) fn advance_burst(&mut self) {
        #[cfg(all(not(test), not(target_arch = "wasm32")))]
        {
            let _ = self.step_inside_verified_engine();
        }
        #[cfg(any(test, target_arch = "wasm32"))]
        {
            self.system.step_n_leapfrog(self.cfg.dt_myr, self.cfg.substeps_per_frame);
        }
    }

    fn render_transport_and_metrics(&mut self, ui: &mut egui::Ui) {
        if let Some(action) = PlaybackTransport::in_thread(
            "galactic_orbits_transport",
            "Galactic Orbit Simulator",
            self.cfg.is_playing,
        )
        .reset_label("↺ Reset Orbits")
        .show(ui)
        {
            match action {
                PlaybackAction::Play => self.cfg.is_playing = true,
                PlaybackAction::Pause => self.cfg.is_playing = false,
                PlaybackAction::Step => {
                    self.cfg.is_playing = false;
                    self.advance_burst();
                }
                PlaybackAction::Reset => {
                    self.cfg.is_playing = false;
                    self.rebuild_system(0);
                }
            }
            self.revision = self.revision.wrapping_add(1).max(1);
        }

        let outer_v = decompose_rotation_curve(4.0 * self.cfg.disk_scale_kpc, &self.system.baryons, &self.system.halo).v_circ_kms;
        render_copyable_metric(ui, "Elapsed Galactic Time", &format!("{:.1} Myr", self.system.elapsed_myr()));
        render_copyable_metric(ui, "Outer v_circ (4 R_d)", &format!("{:.2} km/s", outer_v));
        render_copyable_metric(ui, "Relative Energy Drift |ΔE/E0|", &format!("{:.2e}", self.system.relative_energy_drift()));
        render_copyable_metric(ui, "Angular Momentum Drift |ΔLz/Lz0|", &format!("{:.2e}", self.system.relative_angular_momentum_drift()));
    }

    fn render_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let c = &mut self.cfg;
        let mut rebuild = false;
        ui.heading("Galactic Potential");
        egui::ComboBox::from_label("Galaxy Morphology")
            .selected_text(c.morphology.label())
            .show_ui(ui, |ui| {
                for m in [DwarfMorphology::GeneralIrregular, DwarfMorphology::TypeCode10, DwarfMorphology::TypeCode95To99] {
                    rebuild |= ui.selectable_value(&mut c.morphology, m, m.label()).changed();
                }
            });
        egui::ComboBox::from_label("Dark Halo Profile")
            .selected_text(c.halo_kind.label())
            .show_ui(ui, |ui| {
                for h in [HaloKindChoice::BaryonicOnly, HaloKindChoice::NfwCusp, HaloKindChoice::BurkertCore] {
                    rebuild |= ui.selectable_value(&mut c.halo_kind, h, h.label()).changed();
                }
            });
        rebuild |= ui.add(egui::Slider::new(&mut c.distance_mpc, 0.5..=250.0).text("Distance D (Mpc)")).changed();
        rebuild |= ui.add(egui::Slider::new(&mut c.gas_fraction, 0.0..=0.95).text("HI Gas Fraction")).changed();
        rebuild |= ui.add(egui::Slider::new(&mut c.disk_scale_kpc, 0.2..=5.0).text("Disk Scale R_d (kpc)")).changed();
        rebuild |= ui.add(egui::Slider::new(&mut c.halo_log10_mass, 8.0..=12.0).text("Halo log10(M200 / M_sun)")).changed();
        rebuild |= ui.add(egui::Slider::new(&mut c.halo_scale_kpc, 0.3..=10.0).text("Halo Scale / Core r_s (kpc)")).changed();
        rebuild |= ui.add(egui::Slider::new(&mut c.halo_concentration, 4.0..=25.0).text("Concentration c")).changed();
        ui.separator();
        ui.heading("Symplectic Leapfrog");
        if ui.add(egui::Slider::new(&mut c.dt_myr, 0.1..=10.0).text("Leapfrog Step dt (Myr)")).changed()
            || ui.add(egui::Slider::new(&mut c.substeps_per_frame, 1..=16).text("Substeps / Frame")).changed()
        {
            self.revision = self.revision.wrapping_add(1).max(1);
        }
        rebuild |= ui.add(egui::Slider::new(&mut self.cfg.disk_tracers, 4..=48).text("Disk Tracer Stars")).changed();
        rebuild |= ui.add(egui::Slider::new(&mut self.cfg.tidal_tracers, 0..=24).text("Tidal Stream Stars")).changed();
        rebuild |= ui.add(egui::Slider::new(&mut self.cfg.tidal_tangential_ratio, 0.2..=1.2).text("Stream v_tan / v_circ")).changed();
        rebuild
    }

    fn scale_ring_points(&self, segments: usize) -> Vec<[f64; 2]> {
        let r = self.cfg.halo_scale_kpc;
        (0..=segments)
            .map(|k| {
                let (s, c) = (std::f64::consts::TAU * (k as f64 / segments as f64)).sin_cos();
                [r * c, r * s]
            })
            .collect()
    }

    fn collect_orbital_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let (mut disk_pts, mut stream_pts) = (Vec::new(), Vec::new());
        for p in self.system.particles() {
            match p.kind {
                OrbitParticleKind::DiskTracer => disk_pts.push([p.x_kpc, p.y_kpc]),
                OrbitParticleKind::TidalStream => stream_pts.push([p.x_kpc, p.y_kpc]),
            }
        }
        let mut out = vec![
            ("Disk Tracer Stars".into(), disk_pts),
            ("Halo Scale Radius Ring".into(), self.scale_ring_points(48)),
        ];
        if !stream_pts.is_empty() {
            out.push(("Tidal Stream Particles".into(), stream_pts));
        }
        let mut trail_buf = [[0.0_f64; 2]; TRAIL_CAPACITY];
        for idx in [0usize, self.cfg.disk_tracers] {
            let n = self.system.copy_particle_trail(idx, &mut trail_buf);
            if n >= 2 {
                out.push((format!("Tracer {idx} Orbit Trail"), trail_buf[..n].to_vec()));
            }
        }
        out
    }

    fn render_orbit_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let (mut disk_pts, mut stream_pts) = (Vec::new(), Vec::new());
        for p in self.system.particles() {
            match p.kind {
                OrbitParticleKind::DiskTracer => disk_pts.push([p.x_kpc, p.y_kpc]),
                OrbitParticleKind::TidalStream => stream_pts.push([p.x_kpc, p.y_kpc]),
            }
        }
        let scale_ring = self.scale_ring_points(64);
        Plot::new("galactic_orbits_midplane_plot")
            .data_aspect(1.0)
            .view_aspect(1.35)
            .x_axis_label("Galactocentric x (kpc)")
            .y_axis_label("Galactocentric y (kpc)")
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(Line::new("Halo Scale / Core Radius", PlotPoints::new(scale_ring)).color(palette.info).style(LineStyle::Dashed { length: 6.0 }));
                let mut trail_buf = [[0.0_f64; 2]; TRAIL_CAPACITY];
                for (idx, p) in self.system.particles().iter().enumerate() {
                    if idx % 2 != 0 && p.kind == OrbitParticleKind::DiskTracer {
                        continue;
                    }
                    let n = self.system.copy_particle_trail(idx, &mut trail_buf);
                    if n >= 2 {
                        let col = if p.kind == OrbitParticleKind::DiskTracer { palette.axis_z } else { palette.axis_x };
                        plot_ui.line(Line::new("", PlotPoints::new(trail_buf[..n].to_vec())).color(col).width(1.2_f32));
                    }
                }
                plot_ui.points(Points::new("Disk Tracer Stars", PlotPoints::new(disk_pts)).color(palette.accent).radius(4.0_f32).shape(MarkerShape::Circle));
                if !stream_pts.is_empty() {
                    plot_ui.points(Points::new("Tidal Stream Stars", PlotPoints::new(stream_pts)).color(palette.axis_x).radius(4.5_f32).shape(MarkerShape::Diamond));
                }
            });
    }
}

impl FiniteStateGuard for GalacticOrbitSimulator {
    fn is_finite_state(&self) -> bool {
        let c = &self.cfg;
        [c.distance_mpc, c.gas_fraction, c.disk_scale_kpc, c.halo_log10_mass, c.halo_scale_kpc, c.halo_concentration, c.dt_myr, c.tidal_tangential_ratio]
            .into_iter()
            .all(f64::is_finite)
            && c.dt_myr > 0.0
            && self.system.particles().iter().all(|p| p.x_kpc.is_finite() && p.y_kpc.is_finite() && p.vx_kms.is_finite() && p.vy_kms.is_finite())
    }

    fn auto_scale_step_size(&mut self) {
        self.cfg.dt_myr = (self.cfg.dt_myr * 0.5).clamp(0.1, 10.0);
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    fn clamp_parameters(&mut self) {
        let c = &mut self.cfg;
        c.distance_mpc = clamp_finite(c.distance_mpc, 0.5, 250.0, 12.0);
        c.gas_fraction = clamp_finite(c.gas_fraction, 0.0, 0.95, 0.60);
        c.disk_scale_kpc = clamp_finite(c.disk_scale_kpc, 0.2, 5.0, 0.80);
        c.halo_log10_mass = clamp_finite(c.halo_log10_mass, 8.0, 12.0, 10.1);
        c.halo_scale_kpc = clamp_finite(c.halo_scale_kpc, 0.3, 10.0, 2.4);
        c.halo_concentration = clamp_finite(c.halo_concentration, 4.0, 25.0, 12.0);
        c.dt_myr = clamp_finite(c.dt_myr, 0.1, 10.0, 1.5);
        c.substeps_per_frame = c.substeps_per_frame.clamp(1, 16);
        c.disk_tracers = c.disk_tracers.clamp(4, 48);
        c.tidal_tracers = c.tidal_tracers.clamp(0, 24);
        c.tidal_tangential_ratio = clamp_finite(c.tidal_tangential_ratio, 0.2, 1.2, 0.62);
        self.rebuild_system(0);
    }
}

impl InteractiveTool for GalacticOrbitSimulator {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Galactic Orbit Simulator"
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
    fn step_simulation(&mut self) {
        self.advance_burst();
    }
    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.cfg).ok()
    }
    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: GalacticOrbitsState = serde_json::from_str(state).map_err(|e| format!("Invalid galactic orbit state: {e}"))?;
        let valid_floats = [
            (s.distance_mpc, 0.5, 250.0),
            (s.gas_fraction, 0.0, 0.95),
            (s.disk_scale_kpc, 0.2, 5.0),
            (s.halo_log10_mass, 8.0, 12.0),
            (s.halo_scale_kpc, 0.3, 10.0),
            (s.halo_concentration, 4.0, 25.0),
            (s.dt_myr, 0.1, 10.0),
            (s.tidal_tangential_ratio, 0.2, 1.2),
        ]
        .into_iter()
        .all(|(v, lo, hi)| v.is_finite() && (lo..=hi).contains(&v));
        if !valid_floats || !(1..=16).contains(&s.substeps_per_frame) || !(4..=48).contains(&s.disk_tracers) || s.tidal_tracers > 24 {
            return Err("Galactic orbit parameters out of valid physical bounds".into());
        }
        self.cfg = s;
        self.rebuild_system(16);
        Ok(())
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let datasets = egui_plot::get_last_plot_datasets();
        if datasets.is_empty() { self.collect_orbital_datasets() } else { datasets }
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
        let panel_id = crate::framework::scoped_panel_id(ui, "galactic_orbits_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Playback & Invariants");
                self.render_transport_and_metrics(ui);
                ui.separator();
                if self.render_controls(ui) {
                    self.rebuild_system(8);
                }
            });
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Galactic Disk & Tidal Stream Orbits");
            ui.label("Zero-allocation kick-drift-kick leapfrog integration showing differential spiral winding and rosetta stream precession.");
            self.render_orbit_plot(ui);
        });
    }
}

// [cite:dwarf_galaxy_empirical_dependencies]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "GalacticOrbitSimulator",
        display_name: "Galactic Orbit Simulator",
        domain: "astrophysics",
        tags: &["astrophysics", "orbits", "symplectic", "leapfrog", "dark-matter"],
        build: || Box::new(GalacticOrbitSimulator::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for GalacticOrbitSimulator {
    fn theory_description(&self) -> String {
        format!(
            "Planar galactic orbital dynamics $\\ddot{{\\mathbf{{x}}}} = -\\frac{{v_{{\\text{{circ}}}}^2(r)}}{{r^2}}\\mathbf{{x}}$ integrated via second-order symplectic kick-drift-kick leapfrog ($\\Delta t = {:.2}\\,\\text{{Myr}}$, {} profile) conserving canonical energy $E$ and angular momentum $L_z$.",
            self.cfg.dt_myr,
            self.cfg.halo_kind.label()
        )
    }
    fn phonetic_description(&self) -> String {
        "Symplectic leapfrog dwarf galaxy disk and tidal stream orbit simulator".into()
    }
    fn theory_citation(&self) -> String {
        "Tillaboev, Tadjibaev & Otojanova (2025) (tillaboev2025); Binney & Tremaine (2008), Galactic Dynamics; Burkert (1995)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const PRESET_SPIRAL_WINDING: &str = concat!(
    r#"{"metadata":{"schema_version":"1.0","title":"Differential Spiral Winding","domain":"astrophysics","timestamp":"2026-11-21T00:00:00Z"},"#,
    r#""config":{"distance_mpc":16.0,"morphology":"TypeCode10","gas_fraction":0.60,"disk_scale_kpc":0.85,"halo_kind":"NfwCusp","#,
    r#""halo_log10_mass":10.3,"halo_scale_kpc":2.6,"halo_concentration":13.0,"dt_myr":1.8,"substeps_per_frame":4,"disk_tracers":24,"tidal_tracers":4,"tidal_tangential_ratio":0.70}}"#
);

const PRESET_TIDAL_PRECESSION: &str = concat!(
    r#"{"metadata":{"schema_version":"1.0","title":"Dwarf Tidal Stream Precession","domain":"astrophysics","timestamp":"2026-11-21T00:00:00Z"},"#,
    r#""config":{"distance_mpc":14.0,"morphology":"TypeCode95To99","gas_fraction":0.65,"disk_scale_kpc":0.75,"halo_kind":"BurkertCore","#,
    r#""halo_log10_mass":10.0,"halo_scale_kpc":2.0,"halo_concentration":11.0,"dt_myr":1.5,"substeps_per_frame":5,"disk_tracers":12,"tidal_tracers":12,"tidal_tangential_ratio":0.48}}"#
);

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "galactic_orbits_spiral_winding_starter",
        title: "Differential Spiral Winding",
        description: "Shear two-arm disk tracers into differential spiral arcs inside an NFW dark-matter halo.",
        category: "Galactic Dynamics",
        icon: "🌀",
        domain: "astrophysics",
        tool_name: Some("GalacticOrbitSimulator"),
        preset_generator: || PRESET_SPIRAL_WINDING,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "galactic_orbits_tidal_precession_starter",
        title: "Dwarf Tidal Stream Precession",
        description: "Integrate eccentric satellite stream rosetta orbits inside a cored Burkert (1995) dwarf potential.",
        category: "Galactic Dynamics",
        icon: "☄️",
        domain: "astrophysics",
        tool_name: Some("GalacticOrbitSimulator"),
        preset_generator: || PRESET_TIDAL_PRECESSION,
    }
}
