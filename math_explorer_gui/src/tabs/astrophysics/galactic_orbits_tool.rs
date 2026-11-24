#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::galactic_orbits_relativity::{
    clamp_bh_config, is_valid_bh_config, OrbitLensOverlayConfig,
};
use super::{clamp_finite, HaloKindChoice};
use crate::audio::frame::SonificationFrame;
use crate::audio::mappings::galactic_orbits_frame;
use crate::framework::{
    BrushInteractionMode, BrushMode, ExportCapabilities, FiniteStateGuard, InteractionContext,
    InteractiveTool, ShaderDialect, SpatialBrushKit,
};
use eframe::egui;
use math_explorer::physics::astrophysics::{
    decompose_rotation_curve, CentralBlackHoleConfig, DwarfMorphology, GalacticDiskSystem, Mpc,
    OrbitParticleKind, OrbitPopulationConfig,
};

pub(super) const TRAIL_CAPACITY: usize = 64;

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
    #[serde(default, skip_serializing_if = "CentralBlackHoleConfig::is_default")]
    pub(crate) central_bh: CentralBlackHoleConfig,
    #[serde(default, skip_serializing_if = "OrbitLensOverlayConfig::is_default")]
    pub(crate) lens_overlay: OrbitLensOverlayConfig,
    #[serde(default, skip_serializing_if = "is_false")]
    pub(crate) is_playing: bool,
    #[serde(
        default,
        skip_serializing_if = "crate::audio::AcousticSettings::is_default"
    )]
    pub(crate) audio: crate::audio::AcousticSettings,
}

/// Interactive symplectic leapfrog dwarf-galaxy disk, slingshot, and tidal-stream orbit simulator.
pub struct GalacticOrbitSimulator {
    pub(crate) cfg: GalacticOrbitsState,
    pub(crate) system: GalacticDiskSystem,
    pub(crate) brush_kit: SpatialBrushKit,
    pub(crate) launch_speed_factor: f64,
    pub(super) slingshot_anchor: Option<[f64; 2]>,
    pub(super) slingshot_tip: Option<[f64; 2]>,
    pub(crate) last_plot_id: Option<egui::Id>,
    pub(super) revision: u64,
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
            central_bh: CentralBlackHoleConfig::default(),
            lens_overlay: OrbitLensOverlayConfig::default(),
            is_playing: false,
            audio: crate::audio::AcousticSettings::default(),
        };
        let halo = cfg.halo_kind.build_profile(
            cfg.halo_log10_mass,
            cfg.halo_scale_kpc,
            cfg.halo_concentration,
        );
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
        Self {
            cfg,
            system,
            brush_kit: Self::default_brush_kit(),
            launch_speed_factor: 0.88,
            slingshot_anchor: None,
            slingshot_tip: None,
            last_plot_id: None,
            revision: 1,
        }
    }
}

impl GalacticOrbitSimulator {
    pub(crate) fn default_brush_kit() -> SpatialBrushKit {
        SpatialBrushKit::new(BrushMode::VelocityJet, 0.55)
            .with_interaction_mode(BrushInteractionMode::Pan)
            .with_radius_bounds(0.2, 2.4, 0.2)
            .with_cursor_step(0.35, 0.35)
    }

    pub(crate) fn rebuild_system(&mut self, prewarm_steps: usize) {
        self.revision = self.revision.wrapping_add(1).max(1);
        self.slingshot_anchor = None;
        self.slingshot_tip = None;
        let c = &self.cfg;
        let halo =
            c.halo_kind
                .build_profile(c.halo_log10_mass, c.halo_scale_kpc, c.halo_concentration);
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
        if c.central_bh.enabled {
            self.system.set_central_black_hole(c.central_bh);
            self.system.reseed_orbits(
                c.disk_tracers.max(1),
                c.tidal_tracers,
                c.tidal_tangential_ratio,
            );
        }
        if prewarm_steps > 0 {
            self.system.step_n_leapfrog(self.cfg.dt_myr, prewarm_steps);
        }
    }

    /// Injects a zero-allocation `SlingshotProbe` tracer into the pre-allocated ring buffer.
    pub fn launch_slingshot_probe(&mut self, x_kpc: f64, y_kpc: f64, vx_kms: f64, vy_kms: f64) {
        let max_r = self.plot_domain_extent();
        let px = clamp_finite(x_kpc, -max_r, max_r, self.cfg.disk_scale_kpc);
        let py = clamp_finite(y_kpc, -max_r, max_r, 0.0);
        let vx = clamp_finite(vx_kms, -600.0, 600.0, 0.0);
        let vy = clamp_finite(vy_kms, -600.0, 600.0, 40.0);
        let _ = self.system.launch_tracer_in_place(px, py, vx, vy);
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    /// Launches a tangential/eccentric probe at `(x_kpc, y_kpc)` scaled by `launch_speed_factor`.
    pub fn launch_probe_at_cursor(&mut self, x_kpc: f64, y_kpc: f64) {
        let r = x_kpc.hypot(y_kpc).max(0.15);
        let v_local = decompose_rotation_curve(r, &self.system.baryons, &self.system.halo)
            .v_circ_kms
            .max(12.0);
        let speed = v_local * self.launch_speed_factor.clamp(0.2, 2.2);
        let vx = -(y_kpc / r) * speed - 0.18 * (x_kpc / r) * speed;
        let vy = (x_kpc / r) * speed - 0.18 * (y_kpc / r) * speed;
        self.launch_slingshot_probe(x_kpc, y_kpc, vx, vy);
    }

    /// Builds a zero-allocation polyphonic Doppler [`SonificationFrame`] from active tracers.
    pub fn build_audio_frame(&self) -> SonificationFrame {
        let outer_v = decompose_rotation_curve(
            4.0 * self.cfg.disk_scale_kpc,
            &self.system.baryons,
            &self.system.halo,
        )
        .v_circ_kms;
        galactic_orbits_frame(self.system.particles(), outer_v, self.cfg.disk_scale_kpc)
    }

    pub(super) fn plot_domain_extent(&self) -> f64 {
        (5.0 * self.cfg.disk_scale_kpc + 1.5 * self.cfg.halo_scale_kpc).max(4.0)
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
            self.system
                .step_n_leapfrog(self.cfg.dt_myr, self.cfg.substeps_per_frame);
        }
    }

    pub(super) fn scale_ring_points(&self, segments: usize) -> Vec<[f64; 2]> {
        let r = self.cfg.halo_scale_kpc;
        (0..=segments)
            .map(|k| {
                let (s, c) = (std::f64::consts::TAU * (k as f64 / segments as f64)).sin_cos();
                [r * c, r * s]
            })
            .collect()
    }

    fn collect_orbital_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let (mut disk_pts, mut stream_pts, mut probe_pts) = (Vec::new(), Vec::new(), Vec::new());
        for p in self.system.particles() {
            if p.captured {
                continue;
            }
            match p.kind {
                OrbitParticleKind::DiskTracer => disk_pts.push([p.x_kpc, p.y_kpc]),
                OrbitParticleKind::TidalStream => stream_pts.push([p.x_kpc, p.y_kpc]),
                OrbitParticleKind::SlingshotProbe => probe_pts.push([p.x_kpc, p.y_kpc]),
            }
        }
        let mut out = vec![
            ("Disk Tracer Stars".into(), disk_pts),
            ("Halo Scale Radius Ring".into(), self.scale_ring_points(48)),
        ];
        if !stream_pts.is_empty() {
            out.push(("Tidal Stream Particles".into(), stream_pts));
        }
        if !probe_pts.is_empty() {
            out.push(("Slingshot Probes".into(), probe_pts));
        }
        let mut trail_buf = [[0.0_f64; 2]; TRAIL_CAPACITY];
        for idx in [0usize, self.cfg.disk_tracers] {
            let n = self.system.copy_particle_trail(idx, &mut trail_buf);
            if n >= 2 {
                out.push((format!("Tracer {idx} Orbit Trail"), trail_buf[..n].to_vec()));
            }
        }
        self.append_relativistic_orbit_datasets(&mut out);
        out
    }
}

impl FiniteStateGuard for GalacticOrbitSimulator {
    fn is_finite_state(&self) -> bool {
        let c = &self.cfg;
        [
            c.distance_mpc,
            c.gas_fraction,
            c.disk_scale_kpc,
            c.halo_log10_mass,
            c.halo_scale_kpc,
            c.halo_concentration,
            c.dt_myr,
            c.tidal_tangential_ratio,
        ]
        .into_iter()
        .all(f64::is_finite)
            && c.dt_myr > 0.0
            && c.audio.validate().is_ok()
            && is_valid_bh_config(&c.central_bh)
            && c.lens_overlay.is_valid()
            && self.system.particles().iter().all(|p| {
                p.x_kpc.is_finite()
                    && p.y_kpc.is_finite()
                    && p.vx_kms.is_finite()
                    && p.vy_kms.is_finite()
            })
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
        clamp_bh_config(&mut c.central_bh);
        c.lens_overlay.clamp_in_place();
        c.audio = c.audio.sanitized();
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
    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.brush_kit.focused_plot_id()
    }
    fn on_brush(&mut self, ctx: &InteractionContext) {
        if let Some(stroke) = ctx.brush_stroke {
            self.apply_brush_slingshot_stroke(&stroke, ctx.is_dragging);
        }
    }
    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.cfg).ok()
    }
    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: GalacticOrbitsState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid galactic orbit state: {e}"))?;
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
        s.audio.validate()?;
        if !valid_floats
            || !(1..=16).contains(&s.substeps_per_frame)
            || !(4..=48).contains(&s.disk_tracers)
            || s.tidal_tracers > 24
            || !is_valid_bh_config(&s.central_bh)
            || !s.lens_overlay.is_valid()
        {
            return Err("Galactic orbit parameters out of valid physical bounds".into());
        }
        self.cfg = s;
        self.rebuild_system(16);
        Ok(())
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot().with_audio().with_shader()
    }
    fn sonification_frame(&self) -> Option<SonificationFrame> {
        Some(if self.has_finite_parameters() {
            self.build_audio_frame()
        } else {
            SonificationFrame::silent()
        })
    }
    fn acoustic_settings(&self) -> crate::audio::AcousticSettings {
        self.cfg.audio
    }
    fn set_acoustic_settings(&mut self, settings: crate::audio::AcousticSettings) {
        self.cfg.audio = settings.sanitized();
        self.revision = self.revision.wrapping_add(1).max(1);
    }
    fn export_shader_source(
        &self,
        format: crate::framework::ShaderExportFormat,
    ) -> Result<String, String> {
        match format {
            crate::framework::ShaderExportFormat::Wgsl => self.build_galactic_wgsl_shader(),
            crate::framework::ShaderExportFormat::Glsl => {
                self.build_galactic_glsl_shader(ShaderDialect::WebGl2Es300)
            }
        }
    }
    fn export_glsl_shader(&self, dialect: ShaderDialect) -> Result<String, String> {
        self.build_galactic_glsl_shader(dialect)
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let datasets = egui_plot::get_last_plot_datasets();
        if datasets.is_empty() {
            self.collect_orbital_datasets()
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
            ui.heading("Galactic Disk, Slingshot & Tidal Stream Orbits");
            ui.label("Zero-allocation leapfrog integration with SpatialBrushKit velocity slingshots, rosetta precession, and Doppler sonification.");
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
        tags: &["astrophysics", "orbits", "symplectic", "leapfrog", "dark-matter", "slingshot"],
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
        "Symplectic leapfrog dwarf galaxy disk, slingshot, and tidal stream orbit simulator".into()
    }
    fn theory_citation(&self) -> String {
        "Tillaboev, Tadjibaev & Otojanova (2025) (tillaboev2025); Binney & Tremaine (2008), Galactic Dynamics; Burkert (1995)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}
