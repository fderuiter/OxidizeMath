#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Interactive 2D Cut-and-Project Quasicrystal & Reciprocal Bragg Diffraction Studio.

// [cite:solid_state]

use crate::app::ThemeMode;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolMetadata, ToolStateBounds};
use crate::web_io::PlatformInstant;
use eframe::egui;
use math_explorer::physics::solid_state::quasicrystal::{
    QuasicrystalKind, QuasicrystalParams, QuasicrystalWorkspace,
};
use math_explorer::physics::solid_state::reciprocal::{
    BraggDiffractionWorkspace, BravaisLattice2D, Lattice2DParams,
};
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

#[path = "quasicrystal_studio/canvas.rs"]
mod canvas;
#[path = "quasicrystal_studio/presets.rs"]
pub mod presets;
#[path = "quasicrystal_studio/ui.rs"]
mod ui;

#[cfg(test)]
#[path = "quasicrystal_studio/tests.rs"]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
#[path = "quasicrystal_studio/wasm_tests.rs"]
mod wasm_tests;

const FFT_GRID_SIZE: usize = 256;

const fn default_true() -> bool {
    true
}

const fn default_log_gain() -> f64 {
    4.0
}

/// Selects between aperiodic hyperlattice projection and periodic 2D Bravais lattices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum StudioPatternMode {
    /// Higher-dimensional cut-and-project quasicrystal ($\mathbb{Z}^N \to E_\parallel \oplus E_\perp$).
    #[default]
    Quasicrystal,
    /// Crystallographic 2D Bravais lattice with Wigner-Seitz First Brillouin Zone.
    PeriodicBravais,
}

impl StudioPatternMode {
    /// Human-readable selector label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Quasicrystal => "Aperiodic Quasicrystal (Cut & Project)",
            Self::PeriodicBravais => "Periodic 2D Bravais Lattice",
        }
    }
}

/// High-contrast diffraction intensity colormap options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum DiffractionColormap {
    /// Warm amber-magenta-white X-ray diffraction palette.
    #[default]
    Inferno,
    /// Perceptually uniform emerald-cyan-gold palette.
    Viridis,
    /// High-contrast electron-micrograph monochrome palette.
    Monochrome,
}

impl DiffractionColormap {
    /// Human-readable colormap label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Inferno => "Inferno X-Ray",
            Self::Viridis => "Viridis Spectral",
            Self::Monochrome => "TEM Monochrome",
        }
    }
}

/// Bounded serializable parameter state for [`QuasicrystalStudio`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuasicrystalStudioState {
    /// Active pattern family mode (`Quasicrystal` or `PeriodicBravais`).
    pub mode: StudioPatternMode,
    /// Selected quasicrystal hyperlattice family.
    pub quasicrystal: QuasicrystalKind,
    /// Selected periodic 2D Bravais lattice family.
    pub bravais: BravaisLattice2D,
    /// Primary direct-space edge or lattice constant $a \in [0.4, 4.0]$.
    pub edge_length: f64,
    /// Secondary lattice constant $b \in [0.4, 6.0]$ for oblique/rectangular Bravais nets.
    pub lattice_b: f64,
    /// Inter-axial Bravais angle $\gamma \in [20^\circ, 160^\circ]$.
    pub gamma_deg: f64,
    /// Direct-space circular aperture radius $R \in [12.0, 34.0]$.
    pub aperture_radius: f64,
    /// Perpendicular-space phason shift $w_x \in [-4.0, 4.0]$.
    pub phason_x: f64,
    /// Perpendicular-space phason shift $w_y \in [-4.0, 4.0]$.
    pub phason_y: f64,
    /// Reference perpendicular-space phason $w_{x,0} \in [-4.0, 4.0]$ for flip highlighting.
    pub reference_phason_x: f64,
    /// Reference perpendicular-space phason $w_{y,0} \in [-4.0, 4.0]$ for flip highlighting.
    pub reference_phason_y: f64,
    /// Continuous phason drift speed in $E_\perp$ units per second.
    pub phason_drift_speed: f64,
    /// Continuous phason drift direction angle in degrees.
    pub phason_drift_angle_deg: f64,
    /// Whether phason-flipped prototiles are highlighted in direct space.
    #[serde(default = "default_true")]
    pub highlight_flipped_tiles: bool,
    /// Whether the Wigner-Seitz First Brillouin Zone overlay is drawn on Bravais maps.
    #[serde(default = "default_true")]
    pub show_brillouin_zone: bool,
    /// Whether Bragg ring and peak markers are overlaid on the diffraction map.
    #[serde(default = "default_true")]
    pub show_bragg_rings: bool,
    /// Active diffraction intensity colormap.
    #[serde(default)]
    pub colormap: DiffractionColormap,
    /// Logarithmic dynamic-range compression gain in `[1.0, 8.0]`.
    #[serde(default = "default_log_gain")]
    pub log_gain: f64,
    /// Whether continuous phason drift playback is active.
    #[serde(default)]
    pub is_playing: bool,
}

impl Default for QuasicrystalStudioState {
    fn default() -> Self {
        Self {
            mode: StudioPatternMode::Quasicrystal,
            quasicrystal: QuasicrystalKind::Penrose5D,
            bravais: BravaisLattice2D::Hexagonal,
            edge_length: 1.0,
            lattice_b: 1.45,
            gamma_deg: 72.0,
            aperture_radius: 28.0,
            phason_x: 0.0,
            phason_y: 0.0,
            reference_phason_x: 0.0,
            reference_phason_y: 0.0,
            phason_drift_speed: 0.18,
            phason_drift_angle_deg: 36.0,
            highlight_flipped_tiles: true,
            show_brillouin_zone: true,
            show_bragg_rings: true,
            colormap: DiffractionColormap::Inferno,
            log_gain: 4.0,
            is_playing: false,
        }
    }
}

impl QuasicrystalStudioState {
    /// Validates finite numeric bounds on all configuration parameters.
    pub fn validate(&self) -> Result<(), String> {
        let ok = self.edge_length.is_finite()
            && (0.4..=4.0).contains(&self.edge_length)
            && self.lattice_b.is_finite()
            && (0.4..=6.0).contains(&self.lattice_b)
            && self.gamma_deg.is_finite()
            && (20.0..=160.0).contains(&self.gamma_deg)
            && self.aperture_radius.is_finite()
            && (12.0..=34.0).contains(&self.aperture_radius)
            && self.phason_x.is_finite()
            && (-4.0..=4.0).contains(&self.phason_x)
            && self.phason_y.is_finite()
            && (-4.0..=4.0).contains(&self.phason_y)
            && self.reference_phason_x.is_finite()
            && (-4.0..=4.0).contains(&self.reference_phason_x)
            && self.reference_phason_y.is_finite()
            && (-4.0..=4.0).contains(&self.reference_phason_y)
            && self.phason_drift_speed.is_finite()
            && (0.01..=1.5).contains(&self.phason_drift_speed)
            && self.phason_drift_angle_deg.is_finite()
            && (-180.0..=180.0).contains(&self.phason_drift_angle_deg)
            && self.log_gain.is_finite()
            && (1.0..=8.0).contains(&self.log_gain);
        if ok {
            Ok(())
        } else {
            Err("Quasicrystal studio parameters out of valid finite bounds".into())
        }
    }

    fn sanitize_in_place(&mut self) {
        let clean = |v: f64, fallback: f64, lo: f64, hi: f64| {
            if v.is_finite() {
                v.clamp(lo, hi)
            } else {
                fallback
            }
        };
        self.edge_length = clean(self.edge_length, 1.0, 0.4, 4.0);
        self.lattice_b = clean(self.lattice_b, 1.45, 0.4, 6.0);
        self.gamma_deg = clean(self.gamma_deg, 72.0, 20.0, 160.0);
        self.aperture_radius = clean(self.aperture_radius, 28.0, 12.0, 34.0);
        self.phason_x = clean(self.phason_x, 0.0, -4.0, 4.0);
        self.phason_y = clean(self.phason_y, 0.0, -4.0, 4.0);
        self.reference_phason_x = clean(self.reference_phason_x, 0.0, -4.0, 4.0);
        self.reference_phason_y = clean(self.reference_phason_y, 0.0, -4.0, 4.0);
        self.phason_drift_speed = clean(self.phason_drift_speed, 0.18, 0.01, 1.5);
        self.phason_drift_angle_deg = clean(self.phason_drift_angle_deg, 36.0, -180.0, 180.0);
        self.log_gain = clean(self.log_gain, 4.0, 1.0, 8.0);
    }
}

/// Interactive 2D Quasicrystal Cut-and-Project & Reciprocal Bragg Diffraction Studio.
pub struct QuasicrystalStudio {
    /// Active validated studio state.
    pub state: QuasicrystalStudioState,
    pub(crate) qc_workspace: QuasicrystalWorkspace,
    pub(crate) fft_workspace: BraggDiffractionWorkspace,
    pub(crate) periodic_sites: Vec<[f64; 2]>,
    pub(crate) bragg_texture: Option<egui::TextureHandle>,
    pub(crate) recompute_dirty: bool,
    pub(crate) texture_dirty: bool,
    pub(crate) last_theme: Option<ThemeMode>,
    pub(crate) last_tick: PlatformInstant,
    pub(crate) revision: u64,
}

impl Default for QuasicrystalStudio {
    fn default() -> Self {
        Self {
            state: QuasicrystalStudioState::default(),
            qc_workspace: QuasicrystalWorkspace::new(),
            fft_workspace: BraggDiffractionWorkspace::new(FFT_GRID_SIZE)
                .expect("256x256 Bragg workspace"),
            periodic_sites: Vec::with_capacity(8_192),
            bragg_texture: None,
            recompute_dirty: true,
            texture_dirty: true,
            last_theme: None,
            last_tick: PlatformInstant::now(),
            revision: 1,
        }
    }
}

impl QuasicrystalStudio {
    /// Executes both hyperlattice/Bravais site projection and 256x256 2D `rustfft` with zero heap allocations.
    pub fn recompute_physics(&mut self) {
        match self.state.mode {
            StudioPatternMode::Quasicrystal => {
                let params = QuasicrystalParams {
                    kind: self.state.quasicrystal,
                    edge_length: self.state.edge_length,
                    aperture_radius: self.state.aperture_radius,
                    phason: [self.state.phason_x, self.state.phason_y],
                    reference_phason: [
                        self.state.reference_phason_x,
                        self.state.reference_phason_y,
                    ],
                };
                if self.qc_workspace.project_into(&params).is_ok() {
                    let _ = self.fft_workspace.compute_diffraction_into(
                        self.qc_workspace.positions_xy(),
                        params.edge_length,
                        params.aperture_radius,
                    );
                }
            }
            StudioPatternMode::PeriodicBravais => {
                if let Ok(lat) = self.current_bravais_params() {
                    lat.generate_sites_into(self.state.aperture_radius, &mut self.periodic_sites);
                    let _ = self.fft_workspace.compute_diffraction_into(
                        &self.periodic_sites,
                        lat.a,
                        self.state.aperture_radius,
                    );
                }
            }
        }
        self.recompute_dirty = false;
        self.texture_dirty = true;
    }

    /// Shifts $(w_x, w_y)$ in perpendicular space and recomputes projection + 256x256 FFT with zero allocations.
    pub fn step_phason_zero_alloc(&mut self, delta: [f64; 2]) {
        let wrap = |v: f64| {
            let next = v + 4.0;
            next.rem_euclid(8.0) - 4.0
        };
        self.state.phason_x = wrap(self.state.phason_x + delta[0]);
        self.state.phason_y = wrap(self.state.phason_y + delta[1]);
        self.recompute_physics();
        self.revision = self.revision.wrapping_add(1);
    }

    pub(crate) fn current_bravais_params(&self) -> Result<Lattice2DParams, String> {
        Lattice2DParams::new(
            self.state.bravais,
            self.state.edge_length,
            self.state.lattice_b,
            self.state.gamma_deg.to_radians(),
        )
        .map_err(|e| e.to_string())
    }

    pub(crate) fn active_sites(&self) -> &[[f64; 2]] {
        match self.state.mode {
            StudioPatternMode::Quasicrystal => self.qc_workspace.positions_xy(),
            StudioPatternMode::PeriodicBravais => &self.periodic_sites,
        }
    }

    pub(crate) fn mark_dirty(&mut self) {
        self.recompute_dirty = true;
        self.revision = self.revision.wrapping_add(1);
    }
}

impl InteractiveTool for QuasicrystalStudio {
    fn name(&self) -> &'static str {
        "Quasicrystal & Bragg Studio"
    }

    fn domain(&self) -> &'static str {
        "solid_state"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.revision.hash(&mut h);
        self.state.mode.hash(&mut h);
        self.state.quasicrystal.hash(&mut h);
        self.state.bravais.hash(&mut h);
        self.state.phason_x.to_bits().hash(&mut h);
        self.state.phason_y.to_bits().hash(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        self.state.validate().is_ok()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.state.sanitize_in_place();
        self.recompute_dirty = true;
        self.texture_dirty = true;
        self.revision = self.revision.wrapping_add(1);
        true
    }

    fn state_bounds(&self) -> ToolStateBounds {
        crate::url_state::URL_JSON_BOUNDS
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, raw: &str) -> Result<(), String> {
        self.state_bounds().validate_json(raw)?;
        let parsed: QuasicrystalStudioState = serde_json::from_str(raw)
            .map_err(|e| format!("Invalid quasicrystal state JSON: {e}"))?;
        parsed.validate()?;
        self.state = parsed;
        self.recompute_dirty = true;
        self.texture_dirty = true;
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        let rep = self.fft_workspace.report();
        let stats = self.qc_workspace.stats();
        vec![
            ("empirical_ratio", stats.empirical_ratio),
            ("flipped_tiles", stats.flipped_tile_count as f64),
            ("c4", rep.correlation(4)),
            ("c6", rep.correlation(6)),
            ("c8", rep.correlation(8)),
            ("c10", rep.correlation(10)),
            ("c12", rep.correlation(12)),
        ]
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        Some(self.build_bragg_color_image(ThemeMode::Dark))
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        self.build_export_overlay_primitives()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.collect_export_datasets()
    }

    fn step_simulation(&mut self) {
        let (s, c) = self.state.phason_drift_angle_deg.to_radians().sin_cos();
        let step = 0.04 * self.state.phason_drift_speed;
        self.step_phason_zero_alloc([step * c, step * s]);
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_studio(ui);
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

impl scientific_metadata::theory::TheoryDescribable for QuasicrystalStudio {
    fn theory_description(&self) -> String {
        format!(
            "Projects higher-dimensional integer hyperlattices Z^N -> E_parallel + E_perp for 5D Penrose (golden ratio phi = 1.618034), 4D Ammann-Beenker (silver ratio delta_S = 2.414214), and 6D Stampfli-Gahler (dodecagonal ratio 2 + sqrt(3) = 3.732051) tilings alongside all five periodic 2D Bravais nets, Wigner-Seitz First Brillouin Zones, live perpendicular phason flips (w_x = {:.3}, w_y = {:.3}), and 256x256 2D FFT Bragg structure factors.",
            self.state.phason_x, self.state.phason_y
        )
    }

    fn phonetic_description(&self) -> String {
        "Two-dimensional cut-and-project quasicrystal tilings, perpendicular phason flips, Bravais Brillouin zones, and reciprocal Bragg diffraction".into()
    }

    fn theory_citation(&self) -> String {
        "Shechtman et al. (1984, 10.1103/PhysRevLett.53.1951); de Bruijn (1981); Levine & Steinhardt (1984); Gahler (1988); Baake & Grimm, Aperiodic Order (2013)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    ToolMetadata {
        id: "QuasicrystalStudio",
        display_name: "Quasicrystal & Bragg Studio",
        domain: "solid_state",
        tags: &["quasicrystal", "penrose", "ammann-beenker", "stampfli", "bragg", "brillouin-zone", "phason"],
        build: || Box::new(QuasicrystalStudio::default()),
    }
}
