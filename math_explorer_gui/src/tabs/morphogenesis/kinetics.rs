//! Reaction kinetics adapters, shared worker parameter state, and deterministic noise helpers
//! for [`super::MorphogenesisTool`] and [`super::runner::MorphogenesisKineticsRunner`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::presets::MorphogenesisKineticsMode;
use math_explorer::biology::diffusion::SpatialDiffusion;
use math_explorer::biology::morphogenesis::{
    GrayScottKinetics, ReactionKinetics, SchnakenbergKinetics, SpatialSchnakenbergKinetics,
    TuringSystem,
};
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;

pub(crate) fn apply_uniform_noise<K: ReactionKinetics<2>, D: SpatialDiffusion<2>>(
    system: &mut TuringSystem<2, K, D>,
    count: usize,
    rng: &mut OxidizeRng,
    amp: f64,
) {
    if !amp.is_finite() || amp <= 0.0 {
        return;
    }
    const INV_2_POW_53: f64 = 1.0 / ((1u64 << 53) as f64);
    for idx in 0..count {
        let u_unit = ((rng.next_u64() >> 11) as f64) * INV_2_POW_53;
        let v_unit = ((rng.next_u64() >> 11) as f64) * INV_2_POW_53;
        system.u_mut()[idx] += (2.0 * u_unit - 1.0) * amp;
        system.v_mut()[idx] += (2.0 * v_unit - 1.0) * amp;
    }
}

/// Zero-allocation kinetics adapter switching cleanly between [`SpatialSchnakenbergKinetics`]
/// and [`GrayScottKinetics`] inside [`TuringSystem`].
#[derive(Clone, Debug)]
pub struct MorphogenesisKineticsAdapter {
    /// Active reaction kinetics family.
    pub mode: MorphogenesisKineticsMode,
    /// Schnakenberg activator feed `a`.
    pub a: f64,
    /// Schnakenberg inhibitor feed `b`.
    pub b: f64,
    /// Gray-Scott substrate feed rate `f`.
    pub f: f64,
    /// Gray-Scott catalyst kill rate `k`.
    pub k: f64,
    /// Preallocated spatial `(a, b)` override buffers for Schnakenberg mode.
    pub spatial: SpatialSchnakenbergKinetics,
}

impl MorphogenesisKineticsAdapter {
    /// Creates an adapter with preallocated spatial Schnakenberg mask capacity.
    #[must_use]
    pub fn new(
        mode: MorphogenesisKineticsMode,
        a: f64,
        b: f64,
        f: f64,
        k: f64,
        grid_size: usize,
    ) -> Self {
        Self {
            mode,
            a,
            b,
            f,
            k,
            spatial: SpatialSchnakenbergKinetics::new(a, b, grid_size),
        }
    }

    /// Returns `true` when at least one cell holds an active spatial `(a, b)` override.
    #[cfg(test)]
    #[inline]
    #[must_use]
    pub fn has_spatial_mask(&self) -> bool {
        self.spatial.has_spatial_mask()
    }

    /// Returns the effective Schnakenberg `(a, b)` rates at linear cell `index`.
    #[inline]
    #[must_use]
    pub fn rates_at(&self, index: usize) -> (f64, f64) {
        if self.spatial.is_overridden(index) {
            self.spatial.rates_at(index)
        } else {
            (self.a, self.b)
        }
    }

    /// Assigns a local `(a, b)` reaction override at `index`.
    pub fn set_cell_rates(&mut self, index: usize, a: f64, b: f64) {
        self.spatial.set_cell_rates(index, a, b);
    }

    /// Clears any local override at `index`.
    pub fn clear_cell_override(&mut self, index: usize) {
        self.spatial.clear_cell_override(index);
    }

    /// Clears all spatial overrides in place.
    pub fn clear_mask(&mut self) {
        self.spatial.clear_mask();
    }
}

impl ReactionKinetics<2> for MorphogenesisKineticsAdapter {
    #[inline(always)]
    fn reaction(&self, concentrations: [f64; 2]) -> [f64; 2] {
        match self.mode {
            MorphogenesisKineticsMode::Schnakenberg => {
                SchnakenbergKinetics::new(self.a, self.b).reaction(concentrations)
            }
            MorphogenesisKineticsMode::GrayScott => {
                GrayScottKinetics::new(self.f, self.k).reaction(concentrations)
            }
        }
    }

    #[inline]
    fn reaction_at(&self, index: usize, concentrations: [f64; 2]) -> [f64; 2] {
        match self.mode {
            MorphogenesisKineticsMode::Schnakenberg => {
                let (rate_a, rate_b) = self.rates_at(index);
                SchnakenbergKinetics::new(rate_a, rate_b).reaction(concentrations)
            }
            MorphogenesisKineticsMode::GrayScott => {
                GrayScottKinetics::new(self.f, self.k).reaction(concentrations)
            }
        }
    }
}

/// Shared Gray-Scott mode and `(f, k)` parameters paired with [`MorphogenesisSharedParams`].
#[derive(Clone, Copy, Debug)]
pub struct MorphogenesisKineticsState {
    /// Selected kinetics regime.
    pub mode: MorphogenesisKineticsMode,
    /// Gray-Scott substrate feed rate `f`.
    pub f: f64,
    /// Gray-Scott autocatalyst kill rate `k`.
    pub k: f64,
}

impl Default for MorphogenesisKineticsState {
    fn default() -> Self {
        let gs = GrayScottKinetics::mitosis();
        Self {
            mode: MorphogenesisKineticsMode::Schnakenberg,
            f: gs.f,
            k: gs.k,
        }
    }
}

/// Shared reaction-diffusion parameters synchronized between the UI thread and worker runner.
pub struct MorphogenesisSharedParams {
    /// Schnakenberg activator feed rate `a`.
    pub a: f64,
    /// Schnakenberg inhibitor feed rate `b`.
    pub b: f64,
    /// Primary species diffusion coefficient `D_u`.
    pub d_u: f64,
    /// Secondary species diffusion coefficient `D_v`.
    pub d_v: f64,
    /// Integration step size `dt`.
    pub dt: f64,
    /// Whether high-contrast palette rendering is active.
    pub high_contrast: bool,
    /// Active visual theme mode (`Dark`, `Light`, or `HighContrast`).
    pub theme_mode: crate::app::ThemeMode,
    /// Active counterfactual spatial heatmap mode (`ActiveUb`, `SplitUaUb`, or `DifferenceAbs`).
    pub heatmap_mode: super::counterfactual::MorphogenesisHeatmapMode,
    /// Optional RLE-encoded spatial `(a, b)` kinetics mask (`0` = baseline, `1` = stripe, `2` = spot).
    pub mask_rle: Option<String>,
}
