//! Zero-allocation 2D FitzHugh-Nagumo excitable tissue sheet with virtual electrodes and scars.

use crate::biology::diffusion::FiniteDifference2D;
use crate::biology::morphogenesis::{
    DiffusionCoeff, FitzHughNagumoKinetics, FusedEulerSolver, TuringSystem,
};
use math_commons::math_kernel::types::{Dimension, StepSize, flatten_2d_index};

/// Default spatial grid width and height (`96x96` = `9,216` cells) budgeted for `< 8 ms` WebAssembly slices.
pub const DEFAULT_SHEET_DIM: usize = 96;

/// Timed S2 rectangular stimulus scheduled for S1-S2 vulnerability-window spiral wave induction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScheduledS2Pulse {
    /// Simulation time at which the S2 stimulus fires.
    pub fire_time: f64,
    /// Inclusive starting column index.
    pub x_min: usize,
    /// Exclusive ending column index.
    pub x_max: usize,
    /// Inclusive starting row index.
    pub y_min: usize,
    /// Exclusive ending row index.
    pub y_max: usize,
    /// Target depolarized voltage applied to non-scar cells inside the S2 window.
    pub target_u: f64,
}

/// Pre-allocated 2D FitzHugh-Nagumo excitable tissue sheet.
///
/// Couples [`FitzHughNagumoKinetics`] (`[D_u, 0.0]` diffusion) on [`FiniteDifference2D`] and
/// [`FusedEulerSolver`] with zero-flux insulating scar masks, current-injecting virtual electrodes,
/// and voltage-clamping probes without heap allocations inside [`Self::step`].
///
/// ```rust
/// # use domain_biology::__doctest_facade as math_explorer;
/// use math_explorer::biology::neuroscience::ExcitableSheet2D;
///
/// let mut sheet = ExcitableSheet2D::new_default();
/// sheet.trigger_dual_edge_collision();
/// sheet.step(0.1);
/// assert_eq!(sheet.voltage().len(), 96 * 96);
/// ```
// [cite:biology]
pub struct ExcitableSheet2D {
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    elapsed_time: f64,
    /// Underlying 2-species reaction-diffusion solver (`u` = voltage, `v` = recovery `w`).
    pub system: TuringSystem<2, FitzHughNagumoKinetics, FiniteDifference2D, FusedEulerSolver>,
    prev_u: Vec<f64>,
    pub(super) scar_mask: Vec<u8>,
    pub(super) has_scars: bool,
    pub(super) stimulus: Vec<f64>,
    pub(super) clamp_mask: Vec<u8>,
    pub(super) clamp_voltage: Vec<f64>,
    pending_s2: Option<ScheduledS2Pulse>,
}

impl Default for ExcitableSheet2D {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new_default()
    }
}

impl ExcitableSheet2D {
    /// Creates a `96x96` excitable sheet with default FitzHugh-Nagumo kinetics (`D_u = 0.5`, `dx = dy = 1.0`).
    #[verified_engine::verified]
    pub fn new_default() -> Self {
        Self::new(
            DEFAULT_SHEET_DIM,
            DEFAULT_SHEET_DIM,
            1.0,
            1.0,
            0.5,
            FitzHughNagumoKinetics::default(),
        )
    }

    /// Creates a pre-allocated 2D excitable sheet initialized to uniform resting equilibrium.
    #[verified_engine::verified]
    pub fn new(
        width: usize,
        height: usize,
        dx: f64,
        dy: f64,
        d_u: f64,
        kinetics: FitzHughNagumoKinetics,
    ) -> Self {
        let w = width.max(4);
        let h = height.max(4);
        let safe_dx = dx.max(1e-3);
        let safe_dy = dy.max(1e-3);
        let n = w * h;
        let diffusion = FiniteDifference2D::new(
            Dimension(w),
            Dimension(h),
            StepSize(safe_dx),
            StepSize(safe_dy),
        );
        let system = TuringSystem::new_with_kinetics(
            Dimension(n),
            DiffusionCoeff(d_u.max(0.0)),
            DiffusionCoeff(0.0),
            kinetics,
            diffusion,
        );
        let mut sheet = Self {
            width: w,
            height: h,
            dx: safe_dx,
            dy: safe_dy,
            elapsed_time: 0.0,
            system,
            prev_u: vec![0.0; n],
            scar_mask: vec![0; n],
            has_scars: false,
            stimulus: vec![0.0; n],
            clamp_mask: vec![0; n],
            clamp_voltage: vec![0.0; n],
            pending_s2: None,
        };
        sheet.defibrillate();
        sheet
    }

    /// Grid width in cells.
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Grid height in cells.
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Horizontal grid spacing (`dx`).
    #[verified_engine::verified]
    pub fn dx(&self) -> f64 {
        self.dx
    }

    /// Vertical grid spacing (`dy`).
    #[verified_engine::verified]
    pub fn dy(&self) -> f64 {
        self.dy
    }

    /// Elapsed simulation time since initialization or reset.
    #[verified_engine::verified]
    pub fn elapsed_time(&self) -> f64 {
        self.elapsed_time
    }

    /// Immutable slice of fast membrane potentials ($u$).
    #[verified_engine::verified]
    pub fn voltage(&self) -> &[f64] {
        self.system.u()
    }

    /// Mutable slice of fast membrane potentials ($u$).
    #[verified_engine::verified]
    pub fn voltage_mut(&mut self) -> &mut [f64] {
        self.system.u_mut()
    }

    /// Immutable slice of slow recovery gating variables ($w$).
    #[verified_engine::verified]
    pub fn recovery(&self) -> &[f64] {
        self.system.v()
    }

    /// Mutable slice of slow recovery gating variables ($w$).
    #[verified_engine::verified]
    pub fn recovery_mut(&mut self) -> &mut [f64] {
        self.system.v_mut()
    }

    /// Immutable zero-flux insulating scar mask (`1` = scar, `0` = excitable tissue).
    #[verified_engine::verified]
    pub fn scar_mask(&self) -> &[u8] {
        &self.scar_mask
    }

    /// Mutable zero-flux insulating scar mask (`1` = scar, `0` = excitable tissue).
    #[verified_engine::verified]
    pub fn scar_mask_mut(&mut self) -> &mut [u8] {
        self.has_scars = true;
        &mut self.scar_mask
    }

    /// Immutable slice of local electrode injected currents ($I_{\text{stim}}$).
    #[verified_engine::verified]
    pub fn stimulus(&self) -> &[f64] {
        &self.stimulus
    }

    /// Immutable slice of active voltage-clamp probe flags (`1` = clamped, `0` = free).
    #[verified_engine::verified]
    pub fn clamp_mask(&self) -> &[u8] {
        &self.clamp_mask
    }

    /// Currently scheduled S2 pulse, if an S1-S2 induction protocol is armed.
    #[verified_engine::verified]
    pub fn pending_s2(&self) -> Option<ScheduledS2Pulse> {
        self.pending_s2
    }

    /// Sets a scheduled S2 pulse for S1-S2 spiral induction.
    #[verified_engine::verified]
    pub fn set_pending_s2(&mut self, pulse: Option<ScheduledS2Pulse>) {
        self.pending_s2 = pulse;
    }

    /// Sets the voltage diffusion coefficient $D_u$ (`D_w` remains `0.0`).
    #[verified_engine::verified]
    pub fn set_diffusion_u(&mut self, d_u: f64) {
        self.system.diffusion_coeffs = [d_u.max(0.0), 0.0];
    }

    /// Returns the voltage diffusion coefficient $D_u$.
    #[verified_engine::verified]
    pub fn diffusion_u(&self) -> f64 {
        self.system.diffusion_coeffs[0]
    }

    /// Clears transient current injections and voltage-clamp probes in place.
    #[verified_engine::verified]
    pub fn clear_probes(&mut self) {
        self.stimulus.fill(0.0);
        self.clamp_mask.fill(0);
    }

    /// Clears all insulating scars across the sheet.
    #[verified_engine::verified]
    pub fn clear_scars(&mut self) {
        self.scar_mask.fill(0);
        self.has_scars = false;
    }

    /// Whole-sheet defibrillation shock: resets all cells to resting `(u_rest, w_rest)` while preserving scars.
    #[verified_engine::verified]
    pub fn defibrillate(&mut self) {
        let (u_rest, w_rest) = self.system.kinetics.resting_state();
        self.system.u_mut().fill(u_rest);
        self.system.v_mut().fill(w_rest);
        self.prev_u.fill(u_rest);
        self.stimulus.fill(0.0);
        self.clamp_mask.fill(0);
        self.pending_s2 = None;
    }

    /// Advances the 2D sheet by `dt` with zero heap allocations.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        if dt <= 0.0 || !dt.is_finite() {
            return;
        }
        if self.has_scars {
            self.prev_u.copy_from_slice(self.system.u());
        }
        self.system.step(dt);
        if self.has_scars {
            self.apply_zero_flux_scars_and_probes(dt);
        } else {
            self.apply_probes_fast(dt);
        }
        self.elapsed_time += dt;
        self.fire_scheduled_s2_if_due();
    }

    #[verified_engine::verified]
    fn apply_probes_fast(&mut self, dt: f64) {
        let u_slice = self.system.u_mut();
        for i in 0..u_slice.len() {
            if self.clamp_mask[i] != 0 {
                u_slice[i] = self.clamp_voltage[i];
            } else if self.stimulus[i] != 0.0 {
                u_slice[i] += dt * self.stimulus[i];
            }
        }
    }

    #[verified_engine::verified]
    fn apply_zero_flux_scars_and_probes(&mut self, dt: f64) {
        let w = self.width;
        let h = self.height;
        let d_u = self.system.diffusion_coeffs[0];
        let sx = dt * d_u / (self.dx * self.dx);
        let sy = dt * d_u / (self.dy * self.dy);
        let (u_rest, w_rest) = self.system.kinetics.resting_state();

        for y in 0..h {
            let row = y * w;
            for x in 0..w {
                let idx = row + x;
                if self.scar_mask[idx] != 0 {
                    self.system.u_mut()[idx] = u_rest;
                    self.system.v_mut()[idx] = w_rest;
                } else if self.clamp_mask[idx] != 0 {
                    self.system.u_mut()[idx] = self.clamp_voltage[idx];
                } else {
                    let u_c = self.prev_u[idx];
                    let mut c = 0.0;
                    if x > 0 && self.scar_mask[idx - 1] != 0 {
                        c += sx * (self.prev_u[idx - 1] - u_c);
                    }
                    if x + 1 < w && self.scar_mask[idx + 1] != 0 {
                        c += sx * (self.prev_u[idx + 1] - u_c);
                    }
                    if y > 0 && self.scar_mask[idx - w] != 0 {
                        c += sy * (self.prev_u[idx - w] - u_c);
                    }
                    if y + 1 < h && self.scar_mask[idx + w] != 0 {
                        c += sy * (self.prev_u[idx + w] - u_c);
                    }
                    self.system.u_mut()[idx] += dt * self.stimulus[idx] - c;
                }
            }
        }
    }

    #[verified_engine::verified]
    fn fire_scheduled_s2_if_due(&mut self) {
        if let Some(s2) = self.pending_s2
            && self.elapsed_time >= s2.fire_time
        {
            self.pending_s2 = None;
            let x_end = s2.x_max.min(self.width);
            let y_end = s2.y_max.min(self.height);
            let (_, w_rest) = self.system.kinetics.resting_state();
            for y in s2.y_min.min(y_end)..y_end {
                for x in s2.x_min.min(x_end)..x_end {
                    let idx = flatten_2d_index(x, y, self.width);
                    if self.scar_mask[idx] == 0 && self.system.u()[idx] < 0.0 {
                        self.system.u_mut()[idx] = s2.target_u;
                        self.system.v_mut()[idx] = w_rest;
                    }
                }
            }
        }
    }
}
