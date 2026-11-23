#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Interactive 3D FreeSurfer cortical surface reconstruction, inflation, MRF segmentation,
//! thickness mapping, and cohort GLM statistical morphometry studio.

#[path = "cortical_morphometry/phantom.rs"]
mod phantom;
#[path = "cortical_morphometry/presets.rs"]
mod presets;
#[path = "cortical_morphometry/scene.rs"]
mod scene;
#[path = "cortical_morphometry/ui.rs"]
mod ui;

use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{Camera3D, ExportCapabilities, InteractiveTool, Scene3DBuilder};
use eframe::egui;
use std::collections::HashMap;
use std::hash::{Hash as _, Hasher as _};

/// Vertex scalar field displayed on the 3D cortical manifold.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum CorticalOverlayMode {
    /// Symmetric white-to-pial cortical thickness (`cortical_thickness_map`, mm).
    Thickness,
    /// Multi-subject general linear model contrast (`t_statistic`).
    GlmTStatistic,
    /// Gray-matter posterior probability under the 6-connected Potts MRF prior.
    MrfTissueClass,
    /// Signed sulcal depth (`mm`, positive in sulcal fundi).
    SulcalDepth,
}

impl CorticalOverlayMode {
    /// Ordered list of all selectable cortical overlays.
    pub const ALL: [Self; 4] = [
        Self::Thickness,
        Self::GlmTStatistic,
        Self::MrfTissueClass,
        Self::SulcalDepth,
    ];

    /// Human-readable UI label without raw identifiers.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Thickness => "Cortical Thickness (mm)",
            Self::GlmTStatistic => "GLM Atrophy Contrast (t-stat)",
            Self::MrfTissueClass => "MRF Gray-Matter Posterior",
            Self::SulcalDepth => "Sulcal Folding Depth (mm)",
        }
    }
}

/// Active cortical boundary mesh shown in the 3D viewport and 3D solid export.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum SurfaceDisplayMode {
    /// Outer pial (GM/CSF) cortical boundary.
    Pial,
    /// Inner white-matter (WM/GM) cortical boundary.
    White,
    /// Pial heatmap shell overlaid on inner white-matter reference wireframe.
    DualShell,
}

impl SurfaceDisplayMode {
    /// Ordered list of all selectable cortical boundary modes.
    pub const ALL: [Self; 3] = [Self::Pial, Self::White, Self::DualShell];

    /// Human-readable UI label without raw identifiers.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Pial => "Pial Surface (GM/CSF)",
            Self::White => "White Surface (WM/GM)",
            Self::DualShell => "Dual Pial + White Shell",
        }
    }
}

#[derive(serde::Serialize)]
struct CorticalMorphometrySaveState {
    overlay_mode: CorticalOverlayMode,
    surface_mode: SurfaceDisplayMode,
    inflation: f64,
    u_resolution: usize,
    v_resolution: usize,
    sulcal_depth_mm: f64,
    base_thickness_mm: f64,
    alpha: f64,
    beta: f64,
    sigma: f64,
    learning_rate: f64,
    evolution_steps: usize,
    potts_beta: f64,
    mri_noise_level: f64,
    atrophy_effect_mm: f64,
    collinear_glm_design: bool,
    camera: Camera3D,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CorticalMorphometryState {
    overlay_mode: CorticalOverlayMode,
    surface_mode: SurfaceDisplayMode,
    inflation: f64,
    u_resolution: usize,
    v_resolution: usize,
    sulcal_depth_mm: f64,
    base_thickness_mm: f64,
    alpha: f64,
    beta: f64,
    sigma: f64,
    learning_rate: f64,
    evolution_steps: usize,
    potts_beta: f64,
    mri_noise_level: f64,
    atrophy_effect_mm: f64,
    #[serde(default)]
    collinear_glm_design: bool,
    #[serde(default)]
    camera: Option<Camera3D>,
    #[serde(default)]
    pitch: Option<f32>,
    #[serde(default)]
    yaw: Option<f32>,
    #[serde(default)]
    zoom: Option<f32>,
}

/// Interactive 3D cortical morphometry, deformable surface evolution, Potts MRF, and GLM studio.
pub struct CorticalMorphometryTool {
    /// Active per-vertex scalar heatmap overlay.
    pub overlay_mode: CorticalOverlayMode,
    /// Active cortical boundary surface mode.
    pub surface_mode: SurfaceDisplayMode,
    /// Continuous unfolding interpolation (`0.0` folded to `1.0` smooth ellipsoid).
    pub inflation: f64,
    /// Azimuthal mesh ring vertex count (`12..=48`).
    pub u_resolution: usize,
    /// Meridional latitude ring count (`12..=36`).
    pub v_resolution: usize,
    /// Multi-lobar sulcal folding depth amplitude (mm).
    pub sulcal_depth_mm: f64,
    /// Baseline white-to-pial cortical ribbon thickness (mm).
    pub base_thickness_mm: f64,
    /// Membrane edge-tension regularization weight (`α`).
    pub alpha: f64,
    /// Thin-plate 1-ring Laplacian bending weight (`β`).
    pub beta: f64,
    /// Gaussian MRI gradient smoothing scale (`σ`, mm).
    pub sigma: f64,
    /// Deformable evolution step scale (`η`).
    pub learning_rate: f64,
    /// Number of deformable surface evolution steps (`0..=25`).
    pub evolution_steps: usize,
    /// 6-connected Potts Markov Random Field coupling (`β_MRF`).
    pub potts_beta: f64,
    /// Synthetic T1 MRI additive noise amplitude.
    pub mri_noise_level: f64,
    /// Simulated patient-cohort focal cortical thinning (mm).
    pub atrophy_effect_mm: f64,
    /// Injects collinear predictors to test rank-deficient singular `XᵀX` handling.
    pub collinear_glm_design: bool,
    /// Interactive 3D viewport camera pose.
    pub camera: Camera3D,
    pub(crate) scene_builder: Scene3DBuilder,
    cache: phantom::CorticalPhantomCache,
    /// Structured solver divergence report if non-finite mesh coordinates occur.
    pub diagnostic_report: Option<DiagnosticReport>,
}

impl Default for CorticalMorphometryTool {
    fn default() -> Self {
        let camera = Camera3D::new(0.45, -0.65, 1.0);
        Self {
            overlay_mode: CorticalOverlayMode::Thickness,
            surface_mode: SurfaceDisplayMode::Pial,
            inflation: 0.0,
            u_resolution: 28,
            v_resolution: 20,
            sulcal_depth_mm: 2.4,
            base_thickness_mm: 2.5,
            alpha: 0.35,
            beta: 0.25,
            sigma: 1.2,
            learning_rate: 0.025,
            evolution_steps: 2,
            potts_beta: 1.2,
            mri_noise_level: 8.0,
            atrophy_effect_mm: 0.55,
            collinear_glm_design: false,
            camera,
            scene_builder: Scene3DBuilder::new(camera),
            cache: phantom::CorticalPhantomCache::default(),
            diagnostic_report: None,
        }
    }
}

fn check_range(name: &str, val: f64, min: f64, max: f64) -> Result<(), String> {
    if !val.is_finite() || !(min..=max).contains(&val) {
        Err(format!("{name} out of valid range [{min}, {max}]: {val}"))
    } else {
        Ok(())
    }
}

fn validate_loaded_state(s: &CorticalMorphometryState) -> Result<(), String> {
    check_range("inflation", s.inflation, 0.0, 1.0)?;
    check_range("sulcal_depth_mm", s.sulcal_depth_mm, 0.2, 4.0)?;
    check_range("base_thickness_mm", s.base_thickness_mm, 1.2, 4.0)?;
    check_range("alpha", s.alpha, 0.0, 2.0)?;
    check_range("beta", s.beta, 0.0, 2.0)?;
    check_range("sigma", s.sigma, 0.2, 3.0)?;
    check_range("learning_rate", s.learning_rate, 0.001, 0.2)?;
    check_range("potts_beta", s.potts_beta, 0.0, 3.0)?;
    check_range("mri_noise_level", s.mri_noise_level, 0.0, 30.0)?;
    check_range("atrophy_effect_mm", s.atrophy_effect_mm, 0.0, 1.5)?;
    if !(12..=48).contains(&s.u_resolution) || !(12..=36).contains(&s.v_resolution) {
        return Err("Cortical mesh resolution out of valid range".to_string());
    }
    if s.evolution_steps > 25 {
        return Err("evolution_steps exceeds maximum 25".to_string());
    }
    Ok(())
}

impl FiniteStateGuard for CorticalMorphometryTool {
    fn is_finite_state(&self) -> bool {
        self.has_finite_parameters()
            && self.cache.are_meshes_finite()
            && self.diagnostic_report.is_none()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.diagnostic_report.as_ref()
    }

    fn auto_scale_step_size(&mut self) {
        let scaled = if self.learning_rate.is_finite() {
            (self.learning_rate * 0.5).clamp(0.001, 0.2)
        } else {
            0.0125
        };
        self.learning_rate = scaled;
        self.diagnostic_report = None;
        self.cache.invalidate();
    }

    fn clamp_parameters(&mut self) {
        let clamp_f = |v: f64, lo: f64, hi: f64, def: f64| {
            if v.is_finite() {
                v.clamp(lo, hi)
            } else {
                def
            }
        };
        self.inflation = clamp_f(self.inflation, 0.0, 1.0, 0.0);
        self.u_resolution = self.u_resolution.clamp(12, 48);
        self.v_resolution = self.v_resolution.clamp(12, 36);
        self.sulcal_depth_mm = clamp_f(self.sulcal_depth_mm, 0.2, 4.0, 2.4);
        self.base_thickness_mm = clamp_f(self.base_thickness_mm, 1.2, 4.0, 2.5);
        self.alpha = clamp_f(self.alpha, 0.0, 2.0, 0.35);
        self.beta = clamp_f(self.beta, 0.0, 2.0, 0.25);
        self.sigma = clamp_f(self.sigma, 0.2, 3.0, 1.2);
        self.learning_rate = clamp_f(self.learning_rate, 0.001, 0.2, 0.025);
        self.evolution_steps = self.evolution_steps.min(25);
        self.potts_beta = clamp_f(self.potts_beta, 0.0, 3.0, 1.2);
        self.mri_noise_level = clamp_f(self.mri_noise_level, 0.0, 30.0, 8.0);
        self.atrophy_effect_mm = clamp_f(self.atrophy_effect_mm, 0.0, 1.5, 0.55);
        if !self.camera.is_valid() {
            self.camera = Camera3D::new(0.45, -0.65, 1.0);
        }
        self.diagnostic_report = None;
        self.cache.invalidate();
    }

    fn reset_parameters(&mut self) {
        *self = Self::default();
    }
}

impl InteractiveTool for CorticalMorphometryTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Cortical Morphometry"
    }

    fn domain(&self) -> &'static str {
        "neuroscience"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot().with_3d_solid()
    }

    fn export_colored_mesh_3d(&self) -> Option<Result<oxidize_core::mesh::ColoredMesh, String>> {
        Some(self.build_colored_mesh_3d())
    }

    fn export_mesh_3d(&self) -> Option<Result<oxidize_core::mesh::Mesh, String>> {
        Some(self.build_colored_mesh_3d().map(|cm| cm.mesh))
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.full_state_fingerprint().hash(&mut h);
        self.camera.hash_state(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        (0.0..=1.0).contains(&self.inflation)
            && (12..=48).contains(&self.u_resolution)
            && (12..=36).contains(&self.v_resolution)
            && (0.2..=4.0).contains(&self.sulcal_depth_mm)
            && (1.2..=4.0).contains(&self.base_thickness_mm)
            && (0.0..=2.0).contains(&self.alpha)
            && (0.0..=2.0).contains(&self.beta)
            && (0.2..=3.0).contains(&self.sigma)
            && (0.001..=0.2).contains(&self.learning_rate)
            && self.evolution_steps <= 25
            && (0.0..=3.0).contains(&self.potts_beta)
            && (0.0..=30.0).contains(&self.mri_noise_level)
            && (0.0..=1.5).contains(&self.atrophy_effect_mm)
            && self.camera.is_valid()
            && self.cache.are_meshes_finite()
            && self.diagnostic_report.is_none()
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        self.diagnostic_report.clone()
    }

    fn auto_scale_step_size(&mut self) -> bool {
        FiniteStateGuard::auto_scale_step_size(self);
        true
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("mean_thickness_mm", self.cache.mean_thickness_mm),
            ("total_energy", self.cache.total_energy),
            ("mrf_accuracy", self.cache.mrf_accuracy),
            ("peak_abs_t_stat", self.cache.peak_abs_t_stat),
        ]
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&CorticalMorphometrySaveState {
            overlay_mode: self.overlay_mode,
            surface_mode: self.surface_mode,
            inflation: self.inflation,
            u_resolution: self.u_resolution,
            v_resolution: self.v_resolution,
            sulcal_depth_mm: self.sulcal_depth_mm,
            base_thickness_mm: self.base_thickness_mm,
            alpha: self.alpha,
            beta: self.beta,
            sigma: self.sigma,
            learning_rate: self.learning_rate,
            evolution_steps: self.evolution_steps,
            potts_beta: self.potts_beta,
            mri_noise_level: self.mri_noise_level,
            atrophy_effect_mm: self.atrophy_effect_mm,
            collinear_glm_design: self.collinear_glm_design,
            camera: self.camera,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let loaded: CorticalMorphometryState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid cortical morphometry state: {e}"))?;
        validate_loaded_state(&loaded)?;
        let camera = Camera3D::from_preset_fields(
            loaded.camera,
            loaded.pitch,
            loaded.yaw,
            loaded.zoom,
            Some(Camera3D::new(0.45, -0.65, 1.0)),
        )?;
        self.overlay_mode = loaded.overlay_mode;
        self.surface_mode = loaded.surface_mode;
        self.inflation = loaded.inflation;
        self.u_resolution = loaded.u_resolution;
        self.v_resolution = loaded.v_resolution;
        self.sulcal_depth_mm = loaded.sulcal_depth_mm;
        self.base_thickness_mm = loaded.base_thickness_mm;
        self.alpha = loaded.alpha;
        self.beta = loaded.beta;
        self.sigma = loaded.sigma;
        self.learning_rate = loaded.learning_rate;
        self.evolution_steps = loaded.evolution_steps;
        self.potts_beta = loaded.potts_beta;
        self.mri_noise_level = loaded.mri_noise_level;
        self.atrophy_effect_mm = loaded.atrophy_effect_mm;
        self.collinear_glm_design = loaded.collinear_glm_design;
        self.camera.apply_pose(camera);
        self.diagnostic_report = None;
        self.cache.invalidate();
        Ok(())
    }

    fn step_simulation(&mut self) {
        self.step_deformable_guarded();
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.draw_side_panel(ui);
        self.draw_viewport(ui);
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

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "CorticalMorphometryTool",
        display_name: "Cortical Morphometry",
        domain: "neuroscience",
        tags: &["freesurfer", "mri", "cortex", "mrf", "glm"],
        build: || Box::new(CorticalMorphometryTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for CorticalMorphometryTool {
    fn theory_description(&self) -> String {
        "FreeSurfer cortical surface reconstruction minimizes E_total = E_int + E_ext on a genus-0 triangular manifold, combining edge membrane tension (α), 1-ring Laplacian bending (β), and Gaussian-smoothed MRI boundary gradients (-‖∇G_σ * I‖²). Tissue segmentation couples Gaussian intensity likelihoods with a 6-connected Potts Markov Random Field prior via red-black Iterated Conditional Modes, while symmetric white-to-pial distance maps feed per-vertex General Linear Model t-contrasts.".into()
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some("E_total = (α/2) ∑ ‖x_i - x_j‖² + (β/2) ∑ ‖Δx_i‖² - ∑ ‖∇(G_σ * I)(x_i)‖²".into())
    }

    fn phonetic_description(&self) -> String {
        "Total deformable cortical surface energy equals alpha half edge tension plus beta half discrete Laplacian bending minus squared Gaussian smoothed MRI gradient norm".into()
    }

    fn theory_citation(&self) -> String {
        "Cortical Surface-Based Analysis I: Segmentation and Surface Reconstruction (Dale, Fischl & Sereno, 1999, https://doi.org/10.1006/nimg.1998.0395)".into()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "Deformable Energy".into(),
            "E_int = (α/2) ∑ ‖x_i - x_j‖² + (β/2) ∑ ‖Δx_i‖² balances membrane tension and thin-plate curvature".into(),
        );
        map.insert(
            "Potts MRF".into(),
            "P(c_i = k | I_i, N_i) ∝ N(I_i; μ_k, σ_k²) exp(β_MRF ∑ I[c_j = k]) suppresses isolated MRI voxel noise".into(),
        );
        map.insert(
            "Cortical Thickness & GLM".into(),
            "d(v) = (d(v, S_pial) + d(p_pial, S_white)) / 2 analyzed across subjects via t = cᵀβ / √(σ² cᵀ(XᵀX)⁻¹c)".into(),
        );
        map
    }
}

// [cite:freesurfer]

#[cfg(test)]
#[path = "cortical_morphometry/tests.rs"]
mod tests;
