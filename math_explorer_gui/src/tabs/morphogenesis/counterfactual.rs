//! Counterfactual shadow twin state and spatial difference heatmaps for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub use super::counterfactual_heatmap::render_morphogenesis_heatmap;
use super::runner_twin::clone_turing_system;
use super::sculpt::{enforce_system_mask_generic, MorphoDiffusion};
use super::state::MorphogenesisKinetics;
use crate::framework::{
    field_difference_norms, sanitize_perturbation_delta, DivergenceSnapshot,
    DomainInterventionDelta, SeparationTracker,
};
use math_explorer::biology::diffusion::{MaskedFiniteDifference2D, SpatialDiffusion};
use math_explorer::biology::morphogenesis::{
    GrayScottKinetics, SchnakenbergKinetics, TuringSystem,
};
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

/// Spatial heatmap visualization mode when a counterfactual Turing fork is active.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MorphogenesisHeatmapMode {
    /// Active perturbed activator concentration `u_B(x, y)`.
    #[default]
    ActiveUb,
    /// Side-by-side spatial split `u_A | u_B` separated by a vertical divider seam.
    SplitUaUb,
    /// Pointwise absolute activator difference `|u_B(x, y) - u_A(x, y)|`.
    DifferenceAbs,
}

impl MorphogenesisHeatmapMode {
    /// All selectable spatial heatmap modes in toolbar display order.
    pub const ALL: [Self; 3] = [Self::ActiveUb, Self::SplitUaUb, Self::DifferenceAbs];

    /// Human-readable UI label for segmented mode buttons and accessibility announcements.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ActiveUb => "Active u_B",
            Self::SplitUaUb => "Split u_A | u_B",
            Self::DifferenceAbs => "Difference |u_B - u_A|",
        }
    }
}

/// Compact serializable fork descriptor for `MorphogenesisTool` (`< 320` bytes JSON).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MorphogenesisForkDescriptor {
    /// Step index `k > 0` where the reaction-diffusion field was forked.
    pub fork_step: usize,
    /// Number of lockstep `dt` steps evolved after `fork_step`.
    #[serde(default)]
    pub post_fork_steps: usize,
    /// Frozen baseline activator production rate `a` on `Branch A`.
    pub baseline_a: f64,
    /// Frozen baseline inhibitor production rate `b` on `Branch A`.
    pub baseline_b: f64,
    /// Frozen baseline activator diffusion `D_u` on `Branch A`.
    pub baseline_du: f64,
    /// Frozen baseline inhibitor diffusion `D_v` on `Branch A`.
    pub baseline_dv: f64,
    /// Optional frozen baseline Gray-Scott substrate feed rate `f` on `Branch A`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_f: Option<f64>,
    /// Optional frozen baseline Gray-Scott catalyst kill rate `k` on `Branch A`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_k: Option<f64>,
    /// Spatial activator perturbation amplitude `δu` applied at `t_k` on `Branch B`.
    pub delta_u: f64,
    /// Active spatial heatmap comparison mode.
    #[serde(default)]
    pub heatmap_mode: MorphogenesisHeatmapMode,
}

impl MorphogenesisForkDescriptor {
    /// Validates finiteness and domain bounds of a serialized fork descriptor.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("baseline_a", self.baseline_a, 0.0, 1.0),
            ("baseline_b", self.baseline_b, 0.0, 2.0),
            ("baseline_du", self.baseline_du, 0.01, 5.0),
            ("baseline_dv", self.baseline_dv, 0.005, 200.0),
            ("delta_u", self.delta_u, 0.0, 1.0),
        ];
        if self.fork_step == 0 || self.fork_step > 20_000 || self.post_fork_steps > 20_000 {
            return Err("Morphogenesis fork steps must be in 1..=20000".to_string());
        }
        for (name, val, min, max) in checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "Invalid morphogenesis fork field '{name}' = {val} (expected [{min}, {max}])"
                ));
            }
        }
        for (name, opt) in [
            ("baseline_f", self.baseline_f),
            ("baseline_k", self.baseline_k),
        ] {
            if let Some(val) = opt {
                if !val.is_finite() || !(0.0..=0.2).contains(&val) {
                    return Err(format!(
                        "Invalid morphogenesis fork field '{name}' = {val} (expected [0, 0.2])"
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Structured worker payload carrying active species `u_data`, optional baseline `u_A`, mask roles, and twin metrics.
#[derive(Clone, Debug)]
pub struct MorphogenesisSnapshotData {
    /// Active `Branch B` species field (`u` in Schnakenberg, `v` in Gray-Scott).
    pub u_data: Vec<f64>,
    /// Unperturbed `Branch A` species field `u_A` when forked.
    pub u_a: Option<Vec<f64>>,
    /// Optional row-major `ReactionCellRole` byte grid when any cell is non-`Free`.
    pub mask_roles: Vec<u8>,
    /// Current step index `k`.
    pub step_count: usize,
    /// Live spatial divergence snapshot when forked.
    pub divergence: Option<DivergenceSnapshot>,
}

/// Returns the primary pattern-forming species slice (`u` for Schnakenberg, `v` for Gray-Scott).
#[inline]
#[must_use]
pub fn active_morphogen_slice<D: SpatialDiffusion<2>>(
    sys: &TuringSystem<2, SchnakenbergKinetics, D>,
    mode: MorphogenesisKinetics,
) -> &[f64] {
    match mode {
        MorphogenesisKinetics::Schnakenberg => sys.u(),
        MorphogenesisKinetics::GrayScott => sys.v(),
    }
}

/// Returns a mutable slice of the primary pattern-forming species (`u` or `v`).
#[inline]
pub fn active_morphogen_slice_mut<D: SpatialDiffusion<2>>(
    sys: &mut TuringSystem<2, SchnakenbergKinetics, D>,
    mode: MorphogenesisKinetics,
) -> &mut [f64] {
    match mode {
        MorphogenesisKinetics::Schnakenberg => sys.u_mut(),
        MorphogenesisKinetics::GrayScott => sys.v_mut(),
    }
}

/// Pre-allocated counterfactual shadow twin state for [`super::runner::MorphogenesisRunner`] (zero-alloc `step()`).
pub struct MorphogenesisTwinState<D: SpatialDiffusion<2> = MaskedFiniteDifference2D> {
    /// Frozen baseline `TuringSystem` (`Branch A`) with pre-allocated double buffers and cell mask.
    pub system_a: TuringSystem<2, SchnakenbergKinetics, D>,
    /// Frozen baseline reaction kinetics family (`Schnakenberg` or `GrayScott`).
    pub kinetics_a: MorphogenesisKinetics,
    /// Frozen baseline Gray-Scott `(f, k)` rates on `Branch A`.
    pub gray_scott_a: GrayScottKinetics,
    /// Cloned baseline PRNG state at `t_k`.
    pub rng_a: OxidizeRng,
    /// Step index `k > 0` where the fork was anchored.
    pub fork_step: usize,
    /// Number of lockstep steps taken since `fork_step`.
    pub post_fork_steps: usize,
    /// Spatial activator perturbation amplitude `δu` applied at `t_k`.
    pub delta_u: f64,
    /// Latest spatial Euclidean difference `||u_B - u_A||_2`.
    pub l2_norm: f64,
    /// Latest pointwise supremum difference `||u_B - u_A||_∞`.
    pub linf_norm: f64,
    /// Live separation and logarithmic slope tracker.
    pub tracker: SeparationTracker,
}

impl<D: MorphoDiffusion> MorphogenesisTwinState<D> {
    /// Clones `(fork_step, system_b, rng_b)` into `Branch A` and perturbs `system_b` by `delta_u`.
    #[allow(clippy::too_many_arguments)]
    pub fn fork_from(
        fork_step: usize,
        dt: f64,
        dims: (usize, usize),
        kinetics_b: MorphogenesisKinetics,
        gray_scott_b: GrayScottKinetics,
        system_b: &mut TuringSystem<2, SchnakenbergKinetics, D>,
        rng_b: &mut OxidizeRng,
        delta_u: f64,
    ) -> Self {
        let clean_delta = sanitize_perturbation_delta(delta_u);
        let system_a = clone_turing_system(system_b, dims.0, dims.1);
        let rng_a = rng_b.clone();

        if clean_delta > 0.0 {
            const INV_2_POW_53: f64 = 1.0 / ((1_u64 << 53) as f64);
            let mut fork_rng = OxidizeRng::new(0x5457_494E_464F_524B ^ (fork_step as u64));
            let field_b = active_morphogen_slice_mut(system_b, kinetics_b);
            let center = field_b.len() / 2;
            for (idx, cell) in field_b.iter_mut().enumerate() {
                let jitter = if idx == center {
                    1.0
                } else {
                    2.0 * (((fork_rng.next_u64() >> 11) as f64) * INV_2_POW_53) - 1.0
                };
                *cell += clean_delta * jitter;
            }
        }

        let (l2_norm, linf_norm) = field_difference_norms(
            active_morphogen_slice(&system_a, kinetics_b),
            active_morphogen_slice(system_b, kinetics_b),
        );
        let safe_dt = if dt.is_finite() && dt > 0.0 {
            dt
        } else {
            0.005
        };
        let fork_time = (fork_step as f64) * safe_dt;
        Self {
            system_a,
            kinetics_a: kinetics_b,
            gray_scott_a: gray_scott_b,
            rng_a,
            fork_step,
            post_fork_steps: 0,
            delta_u: clean_delta,
            l2_norm,
            linf_norm,
            tracker: SeparationTracker::new(fork_time, l2_norm),
        }
    }

    /// Clones `MorphogenesisTwinState` outside the verified step loop.
    #[must_use]
    pub fn clone_with_dims(&self, width: usize, height: usize) -> Self {
        Self {
            system_a: clone_turing_system(&self.system_a, width, height),
            kinetics_a: self.kinetics_a,
            gray_scott_a: self.gray_scott_a,
            rng_a: self.rng_a.clone(),
            fork_step: self.fork_step,
            post_fork_steps: self.post_fork_steps,
            delta_u: self.delta_u,
            l2_norm: self.l2_norm,
            linf_norm: self.linf_norm,
            tracker: self.tracker,
        }
    }

    /// Steps frozen baseline `Branch A` by `dt` (with optional 2D CFL substepping) with zero heap allocations.
    #[inline]
    pub fn step_branch_a(
        &mut self,
        dt: f64,
        auto_dt: bool,
        width: usize,
        height: usize,
        noise_amp: f64,
    ) {
        enforce_system_mask_generic(&mut self.system_a, self.kinetics_a, width, height);
        super::runner::step_system_kinetics(
            &mut self.system_a,
            self.kinetics_a,
            &self.gray_scott_a,
            dt,
            auto_dt,
        );
        super::state::apply_uniform_noise(
            &mut self.system_a,
            width * height,
            &mut self.rng_a,
            noise_amp,
        );
        if self.system_a.diffusion.has_active_mask() {
            enforce_system_mask_generic(&mut self.system_a, self.kinetics_a, width, height);
        }
    }

    /// Updates spatial `L2` and `L_∞` norms after a lockstep tick with zero heap allocations.
    #[inline]
    pub fn observe_lockstep(&mut self, step_count: usize, dt: f64, u_b: &[f64]) {
        self.post_fork_steps = step_count.saturating_sub(self.fork_step);
        let ua = active_morphogen_slice(&self.system_a, self.kinetics_a);
        let (l2, linf) = field_difference_norms(ua, u_b);
        self.l2_norm = l2;
        self.linf_norm = linf;
        let safe_dt = if dt.is_finite() && dt > 0.0 {
            dt
        } else {
            0.005
        };
        self.tracker.observe((step_count as f64) * safe_dt, l2);
    }

    /// Builds a live [`DivergenceSnapshot`] reporting spatial `L2` and `L_∞` norms.
    #[must_use]
    pub fn snapshot(&self, step_count: usize, dt: f64) -> DivergenceSnapshot {
        let safe_dt = if dt.is_finite() && dt > 0.0 {
            dt
        } else {
            0.005
        };
        DivergenceSnapshot {
            fork_step: self.fork_step,
            fork_time: self.tracker.fork_time(),
            current_time: (step_count as f64) * safe_dt,
            delta_x: self.delta_u,
            l2_separation: self.l2_norm,
            log_separation_slope: self.tracker.log_separation_slope(),
            peak_separation: self.tracker.peak_separation(),
            primary_delta: Some(DomainInterventionDelta::new(
                "||u_B - u_A||_∞",
                format!("{:.4e}", self.linf_norm),
                self.linf_norm,
                "Maximum pointwise activator difference ||u_B(x,y) - u_A(x,y)||_∞",
            )),
            secondary_delta: Some(DomainInterventionDelta::new(
                "Spatial L2",
                format!("{:.4e}", self.l2_norm),
                self.l2_norm,
                "Spatial Euclidean activator difference ||u_B - u_A||_2",
            )),
        }
    }

    /// Serializes current twin metadata into a [`MorphogenesisForkDescriptor`].
    #[must_use]
    pub fn to_descriptor(
        &self,
        heatmap_mode: MorphogenesisHeatmapMode,
    ) -> MorphogenesisForkDescriptor {
        let is_gs = self.kinetics_a == MorphogenesisKinetics::GrayScott;
        MorphogenesisForkDescriptor {
            fork_step: self.fork_step,
            post_fork_steps: self.post_fork_steps,
            baseline_a: self.system_a.kinetics.a,
            baseline_b: self.system_a.kinetics.b,
            baseline_du: self.system_a.diffusion_coeffs[0],
            baseline_dv: self.system_a.diffusion_coeffs[1],
            baseline_f: is_gs.then_some(self.gray_scott_a.f),
            baseline_k: is_gs.then_some(self.gray_scott_a.k),
            delta_u: self.delta_u,
            heatmap_mode,
        }
    }
}
