#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:solid_state]

#[path = "quasicrystal_diffraction/ui.rs"]
mod ui;

use crate::app::ThemeMode;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_explorer::math_kernel::types::flatten_2d_index;
use math_explorer::physics::solid_state::quasicrystal::{
    QuasicrystalPatchConfig, QuasicrystalSymmetry, QuasicrystalWorkspace,
};
use math_explorer::physics::solid_state::reciprocal::BraggWorkspace;

/// Aperiodic cut-and-project quasicrystals and periodic Bravais slices for kinematic Bragg comparison.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum DiffractionStructure {
    /// 1D golden-ratio Fibonacci chain projected from $\mathbb{Z}^2$.
    Fibonacci1D,
    /// 2D 5-fold pentagonal Penrose tiling projected from $\mathbb{Z}^5$.
    #[default]
    Penrose5Fold,
    /// 2D 8-fold octagonal Ammann-Beenker tiling projected from $\mathbb{Z}^4$.
    AmmannBeenker8Fold,
    /// 2D 10-fold decagonal Penrose tiling projected from $\mathbb{Z}^5$.
    Penrose10Fold,
    /// 2D 12-fold dodecagonal Stampfli tiling projected from $\mathbb{Z}[\zeta_{12}]$.
    Stampfli12Fold,
    /// Periodic 2D simple square lattice (all $(h, k)$ reflections allowed).
    Square2D,
    /// Periodic BCC `(001)` projection ($h + k$ odd reflections extinct).
    Bcc100Slice,
    /// Periodic Centered-Square / FCC `(001)` slice (mixed-parity $(h, k)$ extinct).
    Fcc100Slice,
}

impl DiffractionStructure {
    /// Readable label for UI selectors.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Fibonacci1D => "1D Fibonacci Chain (τ)",
            Self::Penrose5Fold => "2D Penrose Quasicrystal (5-Fold)",
            Self::AmmannBeenker8Fold => "2D Ammann-Beenker (8-Fold)",
            Self::Penrose10Fold => "2D Decagonal Penrose (10-Fold)",
            Self::Stampfli12Fold => "2D Stampfli Quasicrystal (12-Fold)",
            Self::Square2D => "Periodic Simple Cubic (001)",
            Self::Bcc100Slice => "Periodic BCC Projection (h+k Even)",
            Self::Fcc100Slice => "Periodic FCC (001) Slice",
        }
    }

    /// Maps quasicrystal variants to their [`QuasicrystalSymmetry`], or `None` for periodic slices.
    #[must_use]
    pub const fn quasicrystal_symmetry(self) -> Option<QuasicrystalSymmetry> {
        match self {
            Self::Fibonacci1D => Some(QuasicrystalSymmetry::Fibonacci1D),
            Self::Penrose5Fold => Some(QuasicrystalSymmetry::Penrose5Fold),
            Self::AmmannBeenker8Fold => Some(QuasicrystalSymmetry::AmmannBeenker8Fold),
            Self::Penrose10Fold => Some(QuasicrystalSymmetry::Penrose10Fold),
            Self::Stampfli12Fold => Some(QuasicrystalSymmetry::Stampfli12Fold),
            Self::Square2D | Self::Bcc100Slice | Self::Fcc100Slice => None,
        }
    }

    pub(crate) const fn periodic_direct_basis_2d(self) -> [[f64; 2]; 2] {
        match self {
            Self::Bcc100Slice => [[0.5, 0.5], [0.5, -0.5]],
            Self::Fcc100Slice => [[0.5, 0.0], [0.0, 0.5]],
            _ => [[1.0, 0.0], [0.0, 1.0]],
        }
    }
}

/// Serializable URL/preset state for [`QuasicrystalDiffractionTool`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuasicrystalDiffractionState {
    pub structure: DiffractionStructure,
    pub patch_radius: f64,
    pub window_scale: f64,
    pub phason_x: f64,
    pub phason_y: f64,
    pub q_max: f64,
    pub grid_res: usize,
    pub gamma: f64,
    pub show_tile_edges: bool,
    pub show_bz_overlay: bool,
    pub show_perp_window: bool,
}

impl Default for QuasicrystalDiffractionState {
    fn default() -> Self {
        Self {
            structure: DiffractionStructure::Penrose5Fold,
            patch_radius: 3.8,
            window_scale: 1.0,
            phason_x: 0.18,
            phason_y: 0.10,
            q_max: 14.0,
            grid_res: 88,
            gamma: 0.45,
            show_tile_edges: true,
            show_bz_overlay: true,
            show_perp_window: false,
        }
    }
}

impl QuasicrystalDiffractionState {
    pub(crate) fn is_valid(&self) -> bool {
        self.patch_radius.is_finite()
            && (1.2..=5.5).contains(&self.patch_radius)
            && self.window_scale.is_finite()
            && (0.6..=1.35).contains(&self.window_scale)
            && self.phason_x.is_finite()
            && (-1.5..=1.5).contains(&self.phason_x)
            && self.phason_y.is_finite()
            && (-1.5..=1.5).contains(&self.phason_y)
            && self.q_max.is_finite()
            && (4.0..=24.0).contains(&self.q_max)
            && (32..=128).contains(&self.grid_res)
            && self.gamma.is_finite()
            && (0.15..=1.5).contains(&self.gamma)
    }

    pub(crate) fn clamp(&mut self) {
        self.patch_radius = self.patch_radius.clamp(1.2, 5.5);
        self.window_scale = self.window_scale.clamp(0.6, 1.35);
        self.phason_x = self.phason_x.clamp(-1.5, 1.5);
        self.phason_y = self.phason_y.clamp(-1.5, 1.5);
        self.q_max = self.q_max.clamp(4.0, 24.0);
        self.grid_res = self.grid_res.clamp(32, 128);
        self.gamma = self.gamma.clamp(0.15, 1.5);
    }
}

/// Synchronized real-space cut-and-project tiling and reciprocal-space kinematic Bragg explorer.
pub struct QuasicrystalDiffractionTool {
    pub state: QuasicrystalDiffractionState,
    pub(crate) qc_ws: QuasicrystalWorkspace,
    pub(crate) bragg_ws: BraggWorkspace,
    pub(crate) periodic_coords: Vec<[f64; 2]>,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) last_theme: Option<ThemeMode>,
    pub(crate) dirty: bool,
    pub(crate) state_rev: u64,
}

impl Default for QuasicrystalDiffractionTool {
    fn default() -> Self {
        let mut tool = Self {
            state: QuasicrystalDiffractionState::default(),
            qc_ws: QuasicrystalWorkspace::new(),
            bragg_ws: BraggWorkspace::new(128),
            periodic_coords: Vec::with_capacity(512),
            texture: None,
            last_theme: None,
            dirty: true,
            state_rev: 1,
        };
        tool.recompute_physics();
        tool
    }
}

impl QuasicrystalDiffractionTool {
    /// Recomputes real-space atomic positions and reciprocal-space $S(q_x, q_y)$ using preallocated buffers.
    pub fn recompute_physics(&mut self) {
        if let Some(sym) = self.state.structure.quasicrystal_symmetry() {
            let cfg = QuasicrystalPatchConfig {
                symmetry: sym,
                patch_radius: self.state.patch_radius,
                window_scale: self.state.window_scale,
                phason_shift: [self.state.phason_x, self.state.phason_y],
                reference_phason: [0.0, 0.0],
            };
            self.qc_ws.generate_into(&cfg);
            self.bragg_ws.evaluate_grid_verified(
                self.qc_ws.parallel_points(),
                self.state.grid_res,
                self.state.q_max,
                true,
            );
        } else {
            self.populate_periodic_slice();
            self.bragg_ws.evaluate_grid_verified(
                &self.periodic_coords,
                self.state.grid_res,
                self.state.q_max,
                true,
            );
        }
        self.dirty = true;
    }

    fn populate_periodic_slice(&mut self) {
        self.periodic_coords.clear();
        let r_max = self.state.patch_radius.ceil() as i32;
        for ix in -r_max..=r_max {
            for iy in -r_max..=r_max {
                let x = f64::from(ix);
                let y = f64::from(iy);
                if x * x + y * y <= self.state.patch_radius * self.state.patch_radius {
                    self.periodic_coords.push([x, y]);
                    match self.state.structure {
                        DiffractionStructure::Bcc100Slice => {
                            self.periodic_coords.push([x + 0.5, y + 0.5]);
                        }
                        DiffractionStructure::Fcc100Slice => {
                            self.periodic_coords.push([x + 0.5, y + 0.5]);
                            self.periodic_coords.push([x + 0.5, y]);
                            self.periodic_coords.push([x, y + 0.5]);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    /// Renders the current Bragg diffraction map into a [`egui::ColorImage`] using accessible colormaps.
    #[must_use]
    pub fn bragg_color_image(&self, high_contrast: bool) -> egui::ColorImage {
        let n = self.bragg_ws.grid_size().max(1);
        let max_s = self.bragg_ws.peak_intensity().max(1e-12);
        let gamma = self.state.gamma.clamp(0.15, 1.5);
        let mut pixels = vec![egui::Color32::BLACK; n * n];
        let intensities = self.bragg_ws.intensities();
        for iy in 0..n {
            let flipped_row = n - 1 - iy;
            for ix in 0..n {
                let s = intensities[flatten_2d_index(ix, iy, n)];
                let norm = (s / max_s).clamp(0.0, 1.0).powf(gamma);
                let [r, g, b] = if high_contrast {
                    heatmap_color_high_contrast(norm)
                } else {
                    heatmap_color(norm)
                };
                pixels[flatten_2d_index(ix, flipped_row, n)] = egui::Color32::from_rgb(r, g, b);
            }
        }
        egui::ColorImage {
            size: [n, n],
            source_size: egui::Vec2::splat(n as f32),
            pixels,
        }
    }

    pub(crate) fn mark_changed(&mut self) {
        self.state_rev = self.state_rev.wrapping_add(1);
        self.recompute_physics();
    }
}

impl InteractiveTool for QuasicrystalDiffractionTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Quasicrystal & Bragg Diffraction"
    }
    fn domain(&self) -> &'static str {
        "solid_state"
    }
    fn state_revision(&self) -> u64 {
        self.state_rev
    }
    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds {
            max_bytes: 4_096,
            max_depth: 4,
            max_array_len: 16,
        }
    }
    fn has_finite_parameters(&self) -> bool {
        self.state.is_valid()
    }
    fn clamp_parameters(&mut self) -> bool {
        self.state.clamp();
        self.mark_changed();
        true
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }
    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        let hc = self.last_theme == Some(ThemeMode::HighContrast);
        Some(self.bragg_color_image(hc))
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: QuasicrystalDiffractionState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid QuasicrystalDiffraction state: {e}"))?;
        if !parsed.is_valid() {
            return Err("QuasicrystalDiffraction parameters out of finite bounds".into());
        }
        self.state = parsed;
        self.mark_changed();
        Ok(())
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let theme = ThemeMode::from_visuals(parent_ui.visuals());
        if self.last_theme != Some(theme) {
            self.last_theme = Some(theme);
            self.dirty = true;
        }
        self.draw_controls_panel(parent_ui);
        self.draw_central_workspace(parent_ui);
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "QuasicrystalDiffractionTool",
        display_name: "Quasicrystal & Bragg Diffraction",
        domain: "solid_state",
        tags: &["quasicrystal", "penrose", "bragg", "diffraction", "phason"],
        build: || Box::new(QuasicrystalDiffractionTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for QuasicrystalDiffractionTool {
    fn theory_description(&self) -> String {
        format!(
            "Cut-and-project aperiodic order and kinematic Bragg structure factor $S(q) = N^{{-1}}|\\sum_j \\exp(-i q \\cdot r_j)|^2$ for {} (phason shift $(\\phi_x, \\phi_y) = ({:.2}, {:.2})$).",
            self.state.structure.label(),
            self.state.phason_x,
            self.state.phason_y
        )
    }
    fn phonetic_description(&self) -> String {
        "Cut and project quasicrystal geometry, phason flips, and kinematic Bragg diffraction theory".into()
    }
    fn theory_citation(&self) -> String {
        "Metallic Phase with Long-Range Orientational Order and No Translational Symmetry (Shechtman et al., 1984); Quasicrystals: A New Class of Ordered Structures (Levine & Steinhardt, 1984); Quasiperiodic Patterns (Duneau & Katz, 1985)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "CutAndProject".into(),
            "Projection of higher-dimensional Z^D lattice sites falling inside an internal acceptance window".into(),
        );
        map.insert(
            "BraggStructureFactor".into(),
            "Kinematic scattering intensity S(q) preserving sharp rotational peaks across phason flips".into(),
        );
        map
    }
}

const PENROSE_STARTER_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Penrose Quasicrystal & 10-Fold Bragg Map",
    "domain": "solid_state",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "structure": "Penrose5Fold",
    "patch_radius": 3.8,
    "window_scale": 1.0,
    "phason_x": 0.22,
    "phason_y": 0.14,
    "q_max": 14.0,
    "grid_res": 88,
    "gamma": 0.45,
    "show_tile_edges": true,
    "show_bz_overlay": true,
    "show_perp_window": false
  }
}"#;

const STAMPFLI_STARTER_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "12-Fold Stampfli Quasicrystal Diffraction",
    "domain": "solid_state",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "structure": "Stampfli12Fold",
    "patch_radius": 4.0,
    "window_scale": 1.0,
    "phason_x": 0.12,
    "phason_y": -0.08,
    "q_max": 15.0,
    "grid_res": 96,
    "gamma": 0.42,
    "show_tile_edges": true,
    "show_bz_overlay": true,
    "show_perp_window": false
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "quasicrystal_penrose_starter",
        title: "Penrose Quasicrystal & 10-Fold Bragg Map",
        description: "Cut-and-project 5-fold Penrose tiling with live phason vertex flips and sharp 10-fold Bragg rings.",
        category: "Crystallography",
        icon: "✴",
        domain: "solid_state",
        tool_name: Some("QuasicrystalDiffractionTool"),
        preset_generator: || PENROSE_STARTER_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "quasicrystal_stampfli_starter",
        title: "12-Fold Stampfli Quasicrystal Diffraction",
        description: "Forbidden dodecagonal cut-and-project quasicrystal and pseudo-Brillouin zone diffraction map.",
        category: "Crystallography",
        icon: "✳",
        domain: "solid_state",
        tool_name: Some("QuasicrystalDiffractionTool"),
        preset_generator: || STAMPFLI_STARTER_JSON,
    }
}
