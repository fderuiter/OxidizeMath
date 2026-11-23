//! Zero-allocation 1D diffusive Hodgkin-Huxley squid-axon cable model.

use super::model::HodgkinHuxleyModel;
use super::types::{HodgkinHuxleyParameters, HodgkinHuxleyState};
use crate::biology::diffusion::{FiniteDifference1D, SpatialDiffusion};
use math_commons::math_kernel::types::StepSize;
use pure_math::pure_math::analysis::ode::OdeSystem;

/// Default number of spatial compartments in the 1D Hodgkin-Huxley cable (`96`).
pub const DEFAULT_CABLE_LEN: usize = 96;

/// Pre-allocated 1D diffusive Hodgkin-Huxley cable coupling ionic Na+/K+/leak currents with
/// 3-point Neumann spatial diffusion (`FiniteDifference1D`).
///
/// ```rust
/// # use domain_biology::__doctest_facade as math_explorer;
/// use math_explorer::biology::neuroscience::HodgkinHuxleyCable;
///
/// let mut cable = HodgkinHuxleyCable::new_default();
/// cable.pulse_compartment_range(0, 6, 15.0);
/// let mut spiked_distally = false;
/// for _ in 0..250 {
///     cable.step(0.02);
///     if cable.voltage()[12] > 0.0 {
///         spiked_distally = true;
///     }
/// }
/// assert!(spiked_distally);
/// ```
// [cite:biology]
pub struct HodgkinHuxleyCable {
    len: usize,
    dx: f64,
    d_v: f64,
    elapsed_time: f64,
    /// Pre-allocated Hodgkin-Huxley membrane channel model.
    pub model: HodgkinHuxleyModel,
    diffusion: FiniteDifference1D,
    v: Vec<f64>,
    n: Vec<f64>,
    m: Vec<f64>,
    h: Vec<f64>,
    diff_flux: Vec<f64>,
    stimulus: Vec<f64>,
    scar_mask: Vec<u8>,
    clamp_mask: Vec<u8>,
    clamp_voltage: Vec<f64>,
}

impl Default for HodgkinHuxleyCable {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new_default()
    }
}

impl HodgkinHuxleyCable {
    /// Creates a `96`-compartment squid-axon cable at resting state (`-65.0 mV`, `dx = 0.5`, `D_v = 1.2`).
    #[verified_engine::verified]
    pub fn new_default() -> Self {
        Self::new(
            DEFAULT_CABLE_LEN,
            0.5,
            1.2,
            HodgkinHuxleyParameters::default(),
        )
    }

    /// Creates a pre-allocated 1D Hodgkin-Huxley diffusive cable.
    #[verified_engine::verified]
    pub fn new(len: usize, dx: f64, d_v: f64, params: HodgkinHuxleyParameters) -> Self {
        let count = len.max(4);
        let safe_dx = dx.max(1e-3);
        let mut cable = Self {
            len: count,
            dx: safe_dx,
            d_v: d_v.max(0.0),
            elapsed_time: 0.0,
            model: HodgkinHuxleyModel::new(params, 0.0),
            diffusion: FiniteDifference1D::new(StepSize(safe_dx)),
            v: vec![-65.0; count],
            n: vec![0.3177; count],
            m: vec![0.0529; count],
            h: vec![0.5961; count],
            diff_flux: vec![0.0; count],
            stimulus: vec![0.0; count],
            scar_mask: vec![0; count],
            clamp_mask: vec![0; count],
            clamp_voltage: vec![-65.0; count],
        };
        cable.reset_resting();
        cable
    }

    /// Number of compartments in the 1D cable.
    #[verified_engine::verified]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the cable has zero compartments.
    #[verified_engine::verified]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Compartment spacing (`dx`).
    #[verified_engine::verified]
    pub fn dx(&self) -> f64 {
        self.dx
    }

    /// Axial voltage diffusion coefficient (`D_v`).
    #[verified_engine::verified]
    pub fn diffusion_v(&self) -> f64 {
        self.d_v
    }

    /// Sets the axial voltage diffusion coefficient (`D_v`).
    #[verified_engine::verified]
    pub fn set_diffusion_v(&mut self, d_v: f64) {
        self.d_v = d_v.max(0.0);
    }

    /// Elapsed simulation time (ms).
    #[verified_engine::verified]
    pub fn elapsed_time(&self) -> f64 {
        self.elapsed_time
    }

    /// Membrane potentials ($V$, mV) across compartments.
    #[verified_engine::verified]
    pub fn voltage(&self) -> &[f64] {
        &self.v
    }

    /// Sodium activation gating variable ($m$) across compartments.
    #[verified_engine::verified]
    pub fn sodium_m(&self) -> &[f64] {
        &self.m
    }

    /// Sodium inactivation gating variable ($h$) across compartments.
    #[verified_engine::verified]
    pub fn sodium_h(&self) -> &[f64] {
        &self.h
    }

    /// Potassium activation gating variable ($n$) across compartments.
    #[verified_engine::verified]
    pub fn potassium_n(&self) -> &[f64] {
        &self.n
    }

    /// Conduction-block scar mask (`1` = zero-flux block, `0` = excitable axon).
    #[verified_engine::verified]
    pub fn scar_mask(&self) -> &[u8] {
        &self.scar_mask
    }

    /// Resets all cable compartments to resting membrane equilibrium (`-65 mV`), preserving scars.
    #[verified_engine::verified]
    pub fn reset_resting(&mut self) {
        self.v.fill(-65.0);
        self.n.fill(0.3177);
        self.m.fill(0.0529);
        self.h.fill(0.5961);
        self.stimulus.fill(0.0);
        self.clamp_mask.fill(0);
    }

    /// Clears all conduction-block scars along the cable.
    #[verified_engine::verified]
    pub fn clear_scars(&mut self) {
        self.scar_mask.fill(0);
    }

    /// Clears transient injected currents and voltage clamps along the cable.
    #[verified_engine::verified]
    pub fn clear_probes(&mut self) {
        self.stimulus.fill(0.0);
        self.clamp_mask.fill(0);
    }

    /// Depolarizes compartments in `[start, end)` to `target_mv` to initiate a traveling spike.
    #[verified_engine::verified]
    pub fn pulse_compartment_range(&mut self, start: usize, end: usize, target_mv: f64) {
        let hi = end.min(self.len);
        for i in start.min(hi)..hi {
            if self.scar_mask[i] == 0 {
                self.v[i] = target_mv;
            }
        }
    }

    /// Triggers opposing action potentials from both cable ends to demonstrate collision annihilation.
    #[verified_engine::verified]
    pub fn trigger_dual_end_collision(&mut self) {
        self.clear_scars();
        self.reset_resting();
        let band = (self.len / 16).max(3);
        self.pulse_compartment_range(0, band, 10.0);
        self.pulse_compartment_range(self.len - band, self.len, 10.0);
    }

    /// Paints electrode current along `[center - radius, center + radius]`.
    #[verified_engine::verified]
    pub fn paint_brush(&mut self, center: f64, radius: f64, current: f64) {
        let (lo, hi) = self.brush_range(center, radius);
        for i in lo..=hi {
            if self.scar_mask[i] == 0 {
                self.stimulus[i] = current;
            }
        }
    }

    /// Paints or clears a voltage clamp along `[center - radius, center + radius]`.
    #[verified_engine::verified]
    pub fn paint_clamp(&mut self, center: f64, radius: f64, voltage_mv: f64, enabled: bool) {
        let (lo, hi) = self.brush_range(center, radius);
        for i in lo..=hi {
            if self.scar_mask[i] == 0 {
                self.clamp_mask[i] = u8::from(enabled);
                if enabled {
                    self.clamp_voltage[i] = voltage_mv;
                    self.v[i] = voltage_mv;
                }
            }
        }
    }

    /// Paints or erases a zero-flux conduction block along `[center - radius, center + radius]`.
    #[verified_engine::verified]
    pub fn paint_scar(&mut self, center: f64, radius: f64, is_scar: bool) {
        let (lo, hi) = self.brush_range(center, radius);
        for i in lo..=hi {
            self.scar_mask[i] = u8::from(is_scar);
            if is_scar {
                self.v[i] = -65.0;
                self.n[i] = 0.3177;
                self.m[i] = 0.0529;
                self.h[i] = 0.5961;
            }
        }
    }

    /// Advances the 1D Hodgkin-Huxley cable by `dt` (ms) with zero heap allocations.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        if dt <= 0.0 || !dt.is_finite() {
            return;
        }
        let flux_buf = &mut self.diff_flux;
        SpatialDiffusion::<1>::map_diffusion(&self.diffusion, [&self.v], [self.d_v], |i, _, d| {
            flux_buf[i] = d[0];
        });
        let inv_dx2 = 1.0 / (self.dx * self.dx);

        for i in 0..self.len {
            if self.scar_mask[i] != 0 {
                self.v[i] = -65.0;
                continue;
            }
            if self.clamp_mask[i] != 0 {
                self.v[i] = self.clamp_voltage[i];
                continue;
            }
            let cancel = self.scar_flux_cancel_1d(i, inv_dx2);
            let axial_dv = self.diff_flux[i] - cancel;
            self.model.i_ext = self.stimulus[i] + axial_dv;
            let state = HodgkinHuxleyState {
                v: self.v[i],
                n: self.n[i],
                m: self.m[i],
                h: self.h[i],
            };
            let deriv = self.model.derivative(self.elapsed_time, &state);
            self.v[i] += dt * deriv.v;
            self.n[i] = (self.n[i] + dt * deriv.n).clamp(0.0, 1.0);
            self.m[i] = (self.m[i] + dt * deriv.m).clamp(0.0, 1.0);
            self.h[i] = (self.h[i] + dt * deriv.h).clamp(0.0, 1.0);
        }
        self.elapsed_time += dt;
    }

    #[verified_engine::verified]
    fn scar_flux_cancel_1d(&self, i: usize, inv_dx2: f64) -> f64 {
        let mut cancel = 0.0;
        if i > 0 && self.scar_mask[i - 1] != 0 {
            cancel += self.d_v * inv_dx2 * (self.v[i - 1] - self.v[i]);
        }
        if i + 1 < self.len && self.scar_mask[i + 1] != 0 {
            cancel += self.d_v * inv_dx2 * (self.v[i + 1] - self.v[i]);
        }
        cancel
    }

    #[verified_engine::verified]
    fn brush_range(&self, center: f64, radius: f64) -> (usize, usize) {
        let max_idx = (self.len - 1) as f64;
        let r = radius.max(0.5);
        let lo = (center - r).floor().clamp(0.0, max_idx) as usize;
        let hi = (center + r).ceil().clamp(0.0, max_idx) as usize;
        (lo, hi)
    }
}
