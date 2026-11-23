//! State schemas (`#[serde(deny_unknown_fields)]`), [`scientific_metadata::theory::TheoryDescribable`]
//! descriptors, and 4 curated [`crate::starter_gallery::StarterPresetMetadata`] cards for `StochasticPdeTab`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::collections::HashMap;

use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};

use super::kpz_tool::{KpzDimensionMode, KpzPlotView};
use super::spinodal_tool::{PhaseFieldModelChoice, SpinodalBrushAction, SpinodalPlotMode};

pub(crate) const fn default_brush_radius() -> usize {
    4
}

/// Compact `< 4 KiB` URL-safe and preset-safe state for [`super::kpz_tool::KpzInterfaceTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KpzToolState {
    /// Spatial lattice dimensionality (`Dim1D` or `Dim2D`).
    pub dimension: KpzDimensionMode,
    /// Surface tension / diffusive smoothing $\nu \in [0.05, 5.0]$.
    pub nu: f64,
    /// KPZ nonlinear lateral growth coupling $\lambda \in [-8.0, 8.0]$ (`0.0` = Edwards-Wilkinson).
    pub lambda: f64,
    /// Gaussian white-noise strength $\sigma \in [0.0, 4.0]$.
    pub noise_sigma: f64,
    /// Lattice spacing $\Delta x \in [0.25, 4.0]$.
    pub dx: f64,
    /// Explicit Euler-Maruyama step $\Delta t \in [0.002, 0.10]$.
    pub dt: f64,
    /// Lattice length $L$ (`32..=512` in 1D, clamped to `32..=128` in 2D).
    pub system_size: usize,
    /// Substeps executed per animation frame (`1..=16`).
    pub steps_per_frame: usize,
    /// Deterministic `OxidizeRng` seed.
    pub rng_seed: u64,
    /// Active plot view (`RawLogLog` or `FamilyVicsek`).
    #[serde(default)]
    pub plot_view: KpzPlotView,
    /// Interactive mound brush radius in lattice cells (`1..=24`).
    #[serde(default = "default_brush_radius")]
    pub brush_radius: usize,
    /// Whether continuous playback is active.
    #[serde(default)]
    pub playing: bool,
}

impl KpzToolState {
    /// Validates finite bounds and interactive grid caps before applying state.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("nu", self.nu, 0.05, 5.0),
            ("lambda", self.lambda, -8.0, 8.0),
            ("noise_sigma", self.noise_sigma, 0.0, 4.0),
            ("dx", self.dx, 0.25, 4.0),
            ("dt", self.dt, 0.002, 0.10),
        ];
        for (name, val, min, max) in checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "KPZ parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        let max_l = match self.dimension {
            KpzDimensionMode::Dim1D => 512,
            KpzDimensionMode::Dim2D => 128,
        };
        if !(32..=max_l).contains(&self.system_size) {
            return Err(format!(
                "KPZ system_size {} must be within 32..={max_l}",
                self.system_size
            ));
        }
        if !(1..=16).contains(&self.steps_per_frame) {
            return Err(format!(
                "KPZ steps_per_frame {} must be within 1..=16",
                self.steps_per_frame
            ));
        }
        if !(1..=24).contains(&self.brush_radius) {
            return Err(format!(
                "KPZ brush_radius {} must be within 1..=24",
                self.brush_radius
            ));
        }
        Ok(())
    }
}

/// Compact `< 4 KiB` URL-safe and preset-safe state for [`super::spinodal_tool::SpinodalDecompositionTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpinodalToolState {
    /// Hohenberg-Halperin universality model (`AllenCahn` Model A or `CahnHilliard` Model B).
    pub model: PhaseFieldModelChoice,
    /// Bath quench temperature $T \in [0.0, 2.0]$.
    pub temperature: f64,
    /// Critical spinodal temperature $T_c \in [0.2, 2.5]$.
    pub critical_temperature: f64,
    /// Kinetic mobility $M \in [0.05, 5.0]$.
    pub mobility: f64,
    /// Interfacial gradient penalty $\kappa \in [0.05, 4.0]$.
    pub kappa: f64,
    /// Thermal fluctuation strength $\sigma \in [0.0, 0.50]$.
    pub noise_sigma: f64,
    /// Initial spatial mean order parameter $\bar{\phi} \in [-0.65, 0.65]$.
    pub mean_phi: f64,
    /// Semi-implicit time step $\Delta t \in [0.005, 0.25]$.
    pub dt: f64,
    /// Square periodic 2D grid side length (`32..=128`).
    pub grid_size: usize,
    /// Substeps executed per animation frame (`1..=12`).
    pub steps_per_frame: usize,
    /// Deterministic `OxidizeRng` seed.
    pub rng_seed: u64,
    /// Active spectral telemetry plot (`StructureFactor` or `PeakWavenumber`).
    #[serde(default)]
    pub plot_mode: SpinodalPlotMode,
    /// Interactive droplet brush action (`EnrichPositive` or `EnrichNegative`).
    #[serde(default)]
    pub brush_action: SpinodalBrushAction,
    /// Interactive droplet radius (`1..=16`).
    #[serde(default = "default_brush_radius")]
    pub brush_radius: usize,
    /// Whether continuous playback is active.
    #[serde(default)]
    pub playing: bool,
}

impl SpinodalToolState {
    /// Validates finite bounds and interactive grid caps before applying state.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("temperature", self.temperature, 0.0, 2.0),
            ("critical_temperature", self.critical_temperature, 0.2, 2.5),
            ("mobility", self.mobility, 0.05, 5.0),
            ("kappa", self.kappa, 0.05, 4.0),
            ("noise_sigma", self.noise_sigma, 0.0, 0.50),
            ("mean_phi", self.mean_phi, -0.65, 0.65),
            ("dt", self.dt, 0.005, 0.25),
        ];
        for (name, val, min, max) in checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "Spinodal parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        if !(32..=128).contains(&self.grid_size) {
            return Err(format!(
                "Spinodal grid_size {} must be within 32..=128",
                self.grid_size
            ));
        }
        if !(1..=12).contains(&self.steps_per_frame) {
            return Err(format!(
                "Spinodal steps_per_frame {} must be within 1..=12",
                self.steps_per_frame
            ));
        }
        if !(1..=16).contains(&self.brush_radius) {
            return Err(format!(
                "Spinodal brush_radius {} must be within 1..=16",
                self.brush_radius
            ));
        }
        Ok(())
    }
}

/// [`TheoryDescribable`] metadata for Edwards-Wilkinson and KPZ stochastic surface growth.
#[derive(Clone, Debug, Default)]
pub struct KpzTheory;

impl TheoryDescribable for KpzTheory {
    fn theory_description(&self) -> String {
        "Kardar-Parisi-Zhang (KPZ) and Edwards-Wilkinson (EW) stochastic PDEs model kinetic surface roughening with Family-Vicsek scaling W(L, t) = L^alpha f(t / L^z)."
            .to_string()
    }

    fn phonetic_description(&self) -> String {
        "Partial h by partial t equals nu Laplacian h plus one half lambda norm gradient h squared plus sigma white noise eta."
            .to_string()
    }

    fn theory_citation(&self) -> String {
        "Kardar, Parisi & Zhang (1986), DOI: 10.1103/PhysRevLett.56.889; Lam & Shin (1998), DOI: 10.1103/PhysRevE.58.5592"
            .to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "Family-Vicsek Scaling".to_string(),
            "1D EW: (alpha, beta, z) = (1/2, 1/4, 2); 1D KPZ: (alpha, beta, z) = (1/2, 1/3, 3/2) obeying alpha + z = 2."
                .to_string(),
        );
        map.insert(
            "Lam-Shin Stencil".to_string(),
            "Conservative discrete |grad h|^2 estimator preserving exact 1D stationary roughness with tamed Euler-Maruyama stepping."
                .to_string(),
        );
        map
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"\partial_t h = \nu \nabla^2 h + \frac{\lambda}{2}|\nabla h|^2 + \sigma \eta(\mathbf{x}, t)"
                .to_string(),
        )
    }
}

/// [`TheoryDescribable`] metadata for Model A (Allen-Cahn) and Model B (Cahn-Hilliard) phase separation.
#[derive(Clone, Debug, Default)]
pub struct SpinodalTheory;

impl TheoryDescribable for SpinodalTheory {
    fn theory_description(&self) -> String {
        "Stochastic Allen-Cahn (Model A) and Cahn-Hilliard (Model B) phase-field equations evolve Ginzburg-Landau order parameters via Eyre convex-splitting 2D FFT semi-implicit stepping."
            .to_string()
    }

    fn phonetic_description(&self) -> String {
        "Model B partial phi by partial t equals mobility M Laplacian of phi cubed plus T minus T c times phi minus kappa Laplacian phi plus divergence of conserved thermal flux."
            .to_string()
    }

    fn theory_citation(&self) -> String {
        "Cahn & Hilliard (1958), DOI: 10.1063/1.1744102; Hohenberg & Halperin (1977), DOI: 10.1103/RevModPhys.49.435"
            .to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "Conserved Model B".to_string(),
            "Semi-implicit biharmonic FFT solver + staggered face-flux divergence noise preserving mean concentration <phi> to machine precision."
                .to_string(),
        );
        map.insert(
            "Structure Factor S(k, t)".to_string(),
            "Circularly averaged Fourier power spectrum tracking spinodal ring formation and Lifshitz-Slyozov coarsening k_max(t) ~ t^{-1/3}."
                .to_string(),
        );
        map
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"\partial_t \phi = M \nabla^2 \!\left(\phi^3 + (T - T_c)\phi - \kappa \nabla^2 \phi\right) + \nabla \cdot \boldsymbol{\xi}"
                .to_string(),
        )
    }
}

/// Starter preset 1: 1D ballistic KPZ nonlinear surface growth ($\beta = 1/3$).
pub const PRESET_KPZ_BALLISTIC_1D: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "KPZ Interface Growth - Ballistic 1D Scaling",
    "domain": "stochastic_pde",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "dimension": "Dim1D",
    "nu": 0.5,
    "lambda": 3.5,
    "noise_sigma": 1.0,
    "dx": 1.0,
    "dt": 0.025,
    "system_size": 256,
    "steps_per_frame": 6,
    "rng_seed": 101,
    "plot_view": "RawLogLog",
    "brush_radius": 5,
    "playing": false
  }
}"#;

/// Starter preset 2: 1D Edwards-Wilkinson thermal diffusive roughening ($\lambda = 0, \beta = 1/4$).
pub const PRESET_EW_THERMAL_1D: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "KPZ Interface Growth - Edwards-Wilkinson 1D",
    "domain": "stochastic_pde",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "dimension": "Dim1D",
    "nu": 1.0,
    "lambda": 0.0,
    "noise_sigma": 1.0,
    "dx": 1.0,
    "dt": 0.04,
    "system_size": 256,
    "steps_per_frame": 6,
    "rng_seed": 202,
    "plot_view": "FamilyVicsek",
    "brush_radius": 4,
    "playing": false
  }
}"#;

/// Starter preset 3: Conserved Model B Cahn-Hilliard bicontinuous spinodal maze.
pub const PRESET_CAHN_HILLIARD_MAZE: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Spinodal Decomposition - Cahn-Hilliard Maze",
    "domain": "stochastic_pde",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "model": "CahnHilliard",
    "temperature": 0.2,
    "critical_temperature": 1.0,
    "mobility": 1.0,
    "kappa": 0.55,
    "noise_sigma": 0.03,
    "mean_phi": 0.0,
    "dt": 0.06,
    "grid_size": 64,
    "steps_per_frame": 4,
    "rng_seed": 303,
    "plot_mode": "StructureFactor",
    "brush_action": "EnrichPositive",
    "brush_radius": 4,
    "playing": false
  }
}"#;

/// Starter preset 4: Non-conserved Model A Allen-Cahn curvature-driven coarsening.
pub const PRESET_ALLEN_CAHN_DOMAINS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Spinodal Decomposition - Allen-Cahn Domains",
    "domain": "stochastic_pde",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "model": "AllenCahn",
    "temperature": 0.15,
    "critical_temperature": 1.0,
    "mobility": 1.0,
    "kappa": 0.65,
    "noise_sigma": 0.02,
    "mean_phi": 0.0,
    "dt": 0.05,
    "grid_size": 64,
    "steps_per_frame": 4,
    "rng_seed": 404,
    "plot_mode": "PeakWavenumber",
    "brush_action": "EnrichPositive",
    "brush_radius": 5,
    "playing": false
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "spde_kpz_ballistic_1d",
        title: "KPZ Ballistic Scaling (beta = 1/3)",
        description: "Nonlinear lateral interface growth with Lam-Shin stencils and live log-log W(L, t) slope fitting.",
        category: "Stochastic Interfaces",
        icon: "📈",
        domain: "stochastic_pde",
        tool_name: Some("KpzInterfaceTool"),
        preset_generator: || PRESET_KPZ_BALLISTIC_1D,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "spde_ew_thermal_1d",
        title: "Edwards-Wilkinson Roughening (beta = 1/4)",
        description: "Linear diffusive surface roughening (lambda = 0) with Family-Vicsek W / L^alpha collapse.",
        category: "Stochastic Interfaces",
        icon: "〰️",
        domain: "stochastic_pde",
        tool_name: Some("KpzInterfaceTool"),
        preset_generator: || PRESET_EW_THERMAL_1D,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "spde_cahn_hilliard_maze",
        title: "Cahn-Hilliard Spinodal Maze (Model B)",
        description: "Conserved semi-implicit 2D FFT biharmonic spinodal decomposition with radial S(k, t) peak tracking.",
        category: "Phase Separation",
        icon: "🫧",
        domain: "stochastic_pde",
        tool_name: Some("SpinodalDecompositionTool"),
        preset_generator: || PRESET_CAHN_HILLIARD_MAZE,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "spde_allen_cahn_domains",
        title: "Allen-Cahn Curvature Coarsening (Model A)",
        description: "Non-conserved Ginzburg-Landau domain wall relaxation driven by mean interfacial curvature.",
        category: "Phase Separation",
        icon: "🌗",
        domain: "stochastic_pde",
        tool_name: Some("SpinodalDecompositionTool"),
        preset_generator: || PRESET_ALLEN_CAHN_DOMAINS,
    }
}
