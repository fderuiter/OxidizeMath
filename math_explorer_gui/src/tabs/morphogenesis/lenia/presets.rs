//! Species preset metadata, [`TheoryDescribable`] metadata, and Starter Gallery cards for [`super::LeniaTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{LeniaCartographer, LeniaRegime};
use crate::async_sim::atlas::{AtlasTeleportEvent, RegimeCartographer};
use math_explorer::biology::lenia::{
    GrowthParams, LeniaSpecies, MAX_GRID_SIZE, MAX_KERNEL_RADIUS, MAX_KERNEL_RINGS, MIN_GRID_SIZE,
    MIN_KERNEL_RADIUS,
};
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

const EXTINCTION_MASS_FLOOR: f64 = 1e-4;

pub(crate) const fn default_brush_radius() -> f64 {
    4.0
}

/// Compact `< 4 KiB` URL-safe and preset-safe genome state for [`super::LeniaTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeniaToolState {
    /// Concentric kernel outer radius `R` (`4.0..=32.0`).
    pub radius: f64,
    /// Gaussian growth center `mu` (`0.01..=0.60`).
    pub mu: f64,
    /// Gaussian growth width `sigma` (`0.002..=0.15`).
    pub sigma: f64,
    /// Forward-Euler step size `dt` (`0.005..=0.50`).
    pub dt: f64,
    /// Concentric shell weights (`1..=4` entries in `0.0..=1.0`).
    pub ring_weights: Vec<f64>,
    /// Periodic square grid side length (`32..=256`).
    pub grid_size: usize,
    /// Selected species genome template.
    pub species: LeniaSpeciesPreset,
    /// Deterministic PRNG seed for initial perturbation and stamping.
    pub rng_seed: u64,
    /// Density perturbation amplitude (`0.0..=0.35`).
    pub perturbation: f64,
    /// Interactive brush disc radius (`1.0..=24.0`).
    #[serde(default = "default_brush_radius")]
    pub brush_radius: f64,
    /// Active brush action (`Deposit` or `Slice`).
    #[serde(default)]
    pub brush_mode: LeniaBrushMode,
    /// Whether continuous stepping is actively running.
    #[serde(default)]
    pub playing: bool,
}

impl LeniaToolState {
    /// Enforces `URL_JSON_BOUNDS`-compatible domain limits (`grid_size` in `32..=256`, `rings` in `1..=4`).
    pub fn validate(&self) -> Result<(), String> {
        let scalar_bounds = [
            ("radius", self.radius, MIN_KERNEL_RADIUS, MAX_KERNEL_RADIUS),
            ("mu", self.mu, 0.01, 0.60),
            ("sigma", self.sigma, 0.002, 0.15),
            ("dt", self.dt, 0.005, 0.50),
            ("perturbation", self.perturbation, 0.0, 0.35),
            ("brush_radius", self.brush_radius, 1.0, 24.0),
        ];
        for (name, value, min, max) in scalar_bounds {
            if !value.is_finite() || value < min || value > max {
                return Err(format!(
                    "Lenia parameter '{name}' = {value} must be finite within [{min}, {max}]"
                ));
            }
        }
        if !(MIN_GRID_SIZE..=MAX_GRID_SIZE).contains(&self.grid_size) {
            return Err(format!(
                "Lenia grid_size {} must be within {}..={}",
                self.grid_size, MIN_GRID_SIZE, MAX_GRID_SIZE
            ));
        }
        self.validate_rings()
    }

    fn validate_rings(&self) -> Result<(), String> {
        if self.ring_weights.is_empty() || self.ring_weights.len() > MAX_KERNEL_RINGS {
            return Err(format!(
                "Lenia ring_weights length {} must be within 1..={}",
                self.ring_weights.len(),
                MAX_KERNEL_RINGS
            ));
        }
        let mut positive_sum = 0.0;
        for (idx, &w) in self.ring_weights.iter().enumerate() {
            if !w.is_finite() || !(0.0..=1.0).contains(&w) {
                return Err(format!(
                    "Lenia ring_weights[{idx}] = {w} must be finite within [0.0, 1.0]"
                ));
            }
            positive_sum += w;
        }
        if positive_sum <= 1e-6 {
            return Err("At least one Lenia ring weight must exceed 0".to_string());
        }
        Ok(())
    }
}

/// JSON preset envelope for canonical `Orbium` glider in Lenia Studio.
pub const PRESET_LENIA_ORBIUM: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Lenia Studio - Orbium Glider",
    "domain": "biology",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "radius": 13.0,
    "mu": 0.15,
    "sigma": 0.015,
    "dt": 0.1,
    "ring_weights": [1.0],
    "grid_size": 128,
    "species": "Orbium",
    "rng_seed": 42,
    "perturbation": 0.0,
    "brush_radius": 4.0,
    "brush_mode": "Deposit",
    "playing": true
  }
}"#;

/// JSON preset envelope for multi-ring `Geminium` dividing organism in Lenia Studio.
pub const PRESET_LENIA_GEMINIUM: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Lenia Studio - Geminium Twin",
    "domain": "biology",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "radius": 16.0,
    "mu": 0.24,
    "sigma": 0.036,
    "dt": 0.1,
    "ring_weights": [1.0, 0.55, 0.75],
    "grid_size": 128,
    "species": "Geminium",
    "rng_seed": 42,
    "perturbation": 0.01,
    "brush_radius": 4.0,
    "brush_mode": "Deposit",
    "playing": true
  }
}"#;

/// JSON preset envelope for three-ring `Hydrogeminium` swimmer in Lenia Studio.
pub const PRESET_LENIA_HYDROGEMINIUM: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Lenia Studio - Hydrogeminium Swimmer",
    "domain": "biology",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "radius": 15.0,
    "mu": 0.26,
    "sigma": 0.036,
    "dt": 0.1,
    "ring_weights": [0.5, 1.0, 0.667],
    "grid_size": 128,
    "species": "Hydrogeminium",
    "rng_seed": 42,
    "perturbation": 0.01,
    "brush_radius": 4.0,
    "brush_mode": "Deposit",
    "playing": true
  }
}"#;

/// Serializable species enum mirroring [`LeniaSpecies`] for URL and JSON preset schemas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LeniaSpeciesPreset {
    /// Single-ring self-propelling glider (`Orbium unicaudatus`).
    #[default]
    Orbium,
    /// Multi-ring twin-lobe dividing organism (`Geminium`).
    Geminium,
    /// Three-ring rotating/wobbling swimmer (`Hydrogeminium natans`).
    Hydrogeminium,
    /// Single-ring shield soliton (`Scutium`).
    Scutium,
}

impl LeniaSpeciesPreset {
    /// Ordered list of all species options exposed in the studio UI.
    pub const ALL: [LeniaSpeciesPreset; 4] = [
        Self::Orbium,
        Self::Geminium,
        Self::Hydrogeminium,
        Self::Scutium,
    ];

    /// Converts to the underlying `domain_biology` [`LeniaSpecies`].
    pub const fn to_domain(self) -> LeniaSpecies {
        match self {
            Self::Orbium => LeniaSpecies::Orbium,
            Self::Geminium => LeniaSpecies::Geminium,
            Self::Hydrogeminium => LeniaSpecies::Hydrogeminium,
            Self::Scutium => LeniaSpecies::Scutium,
        }
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        self.to_domain().display_name()
    }
}

/// Interactive brush mode for painting or slicing organisms on the periodic Lenia canvas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LeniaBrushMode {
    /// Deposits continuous cell mass onto the grid.
    #[default]
    Deposit,
    /// Slices or erases cell mass (e.g., dissecting an `Orbium` tail to observe self-healing).
    Slice,
}

/// [`TheoryDescribable`] descriptor for Lenia spectral cellular automata.
#[derive(Clone, Debug, Default)]
pub struct LeniaTheory {
    growth: GrowthParams,
}

impl TheoryDescribable for LeniaTheory {
    fn theory_description(&self) -> String {
        "Lenia generalizes Conway's Game of Life to continuous space, time, and states via FFT circular ring convolutions and Gaussian growth niches."
            .to_string()
    }

    fn phonetic_description(&self) -> String {
        self.growth.phonetic_description()
    }

    fn theory_citation(&self) -> String {
        self.growth.theory_citation()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = self.growth.available_descriptions();
        map.insert(
            "kernel_shells".to_string(),
            "Concentric exponential bump shells K(r) normalized to unit L1 mass and convolved via 2D FFT."
                .to_string(),
        );
        map
    }

    fn theory_parameters(
        &self,
    ) -> HashMap<String, scientific_metadata::theory::ParameterConstraint> {
        use scientific_metadata::theory::ParameterConstraint;
        let mut map = self.growth.theory_parameters();
        map.insert(
            "radius".to_string(),
            ParameterConstraint {
                min: MIN_KERNEL_RADIUS,
                max: MAX_KERNEL_RADIUS,
                step: 0.5,
            },
        );
        map.insert(
            "dt".to_string(),
            ParameterConstraint {
                min: 0.005,
                max: 0.50,
                step: 0.005,
            },
        );
        map.insert(
            "perturbation".to_string(),
            ParameterConstraint {
                min: 0.0,
                max: 0.35,
                step: 0.01,
            },
        );
        map.insert(
            "brush_radius".to_string(),
            ParameterConstraint {
                min: 1.0,
                max: 24.0,
                step: 0.5,
            },
        );
        map
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"A_{t+\Delta t} = \operatorname{clamp}\!\left(A_t + \Delta t \, G\!\left(\mathcal{F}^{-1}(\mathcal{F}(A_t)\odot \hat{K});\,\mu,\sigma\right),\,0,\,1\right)"
                .to_string(),
        )
    }
}

impl super::LeniaTool {
    pub(super) fn sanitize_rings_and_field(&mut self) {
        if self.ring_weights.is_empty() {
            self.ring_weights.push(1.0);
        }
        self.ring_weights.truncate(MAX_KERNEL_RINGS);
        for w in &mut self.ring_weights {
            *w = if w.is_finite() {
                w.clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
        if self.ring_weights.iter().copied().sum::<f64>() <= 1e-6 {
            self.ring_weights[0] = 1.0;
        }
        if !self.engine.state().iter().all(|v| v.is_finite()) {
            for cell in self.engine.state_mut() {
                *cell = if cell.is_finite() {
                    cell.clamp(0.0, 1.0)
                } else {
                    0.0
                };
            }
            self.engine.refresh_telemetry();
        }
    }

    /// Returns `true` when current `(mu, sigma, radius, dt)` lies inside a viable soliton or colony regime.
    #[must_use]
    pub fn is_current_niche_viable(&self) -> bool {
        let theory = self.theory.theory_parameters();
        let cart = LeniaCartographer::new(
            &theory,
            ("mu", "sigma"),
            self.species,
            self.radius,
            &self.ring_weights,
            GrowthParams::new(self.mu, self.sigma),
            self.dt,
        );
        let cell = cart.evaluate_cell(self.mu, self.sigma);
        matches!(
            LeniaRegime::from_u8(cell.regime),
            LeniaRegime::StableSoliton | LeniaRegime::ExpandingWave
        )
    }

    /// Re-seeds the active species genome automatically when an extinct grid (`total_mass <= 1e-4`)
    /// enters a viable soliton niche.
    pub fn reseed_if_extinct_entering_viable(&mut self) -> bool {
        if self.engine.telemetry().total_mass <= EXTINCTION_MASS_FLOOR
            && self.is_current_niche_viable()
        {
            self.reseed_current_genome();
            return true;
        }
        false
    }

    #[cfg(test)]
    pub(crate) fn is_finite_and_extinct_for_test(&self) -> bool {
        let mass = self.engine.telemetry().total_mass;
        mass.is_finite() && mass <= EXTINCTION_MASS_FLOOR
    }

    pub(crate) fn lenia_atlas_context_sig(&self, axis_keys: (&'static str, &'static str)) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.species.hash(&mut h);
        self.dt.to_bits().hash(&mut h);
        self.ring_weights.len().hash(&mut h);
        for &w in &self.ring_weights {
            w.to_bits().hash(&mut h);
        }
        for (name, val) in [
            ("mu", self.mu),
            ("sigma", self.sigma),
            ("radius", self.radius),
        ] {
            if name != axis_keys.0 && name != axis_keys.1 {
                name.hash(&mut h);
                val.to_bits().hash(&mut h);
            }
        }
        h.finish().max(1)
    }

    pub(crate) fn active_lenia_atlas_coords(
        &self,
        axis_keys: (&'static str, &'static str),
    ) -> (f64, f64) {
        let get = |k: &str| match k {
            "mu" => self.mu,
            "sigma" => self.sigma,
            "radius" => self.radius,
            _ => 0.0,
        };
        (get(axis_keys.0), get(axis_keys.1))
    }

    pub(crate) fn build_lenia_cartographer(
        &self,
        axis_keys: (&'static str, &'static str),
    ) -> std::sync::Arc<dyn crate::async_sim::atlas::RegimeCartographer> {
        let theory = self.theory.theory_parameters();
        std::sync::Arc::new(LeniaCartographer::new(
            &theory,
            axis_keys,
            self.species,
            self.radius,
            &self.ring_weights,
            GrowthParams::new(self.mu, self.sigma),
            self.dt,
        ))
    }

    pub(crate) fn handle_lenia_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            match key {
                "mu" => self.mu = val.clamp(0.01, 0.60),
                "sigma" => self.sigma = val.clamp(0.002, 0.15),
                "radius" => self.radius = val.clamp(4.0, 32.0),
                _ => {}
            }
        }
        self.diagnostic = None;
        self.sync_engine_parameters();
        if !self.reseed_if_extinct_entering_viable() {
            self.refresh_pixels_in_place();
            self.user_mutation_nonce = self.user_mutation_nonce.wrapping_add(1);
        }
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "lenia_orbium",
        title: "Lenia Orbium Glider",
        description: "A_{t+dt} = clamp(A_t + dt·G(A_t * K)): Self-propelling continuous soliton glider on a 2D periodic torus.",
        category: "Artificial Life",
        icon: "🛸",
        domain: "biology",
        tool_name: Some("LeniaTool"),
        preset_generator: || PRESET_LENIA_ORBIUM,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "lenia_geminium",
        title: "Lenia Geminium Twin",
        description: "Three-ring spectral kernel stabilizing twin-lobed continuous organisms and lobe fission.",
        category: "Artificial Life",
        icon: "🧬",
        domain: "biology",
        tool_name: Some("LeniaTool"),
        preset_generator: || PRESET_LENIA_GEMINIUM,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "lenia_hydrogeminium",
        title: "Lenia Hydrogeminium Swimmer",
        description: "Multi-shell [0.5, 1.0, 0.667] concentric kernel driving rotating and wobbling continuous lifeforms.",
        category: "Artificial Life",
        icon: "🌊",
        domain: "biology",
        tool_name: Some("LeniaTool"),
        preset_generator: || PRESET_LENIA_HYDROGEMINIUM,
    }
}
