//! Zero-allocation `#[wasm_bindgen]` SDE and BAOAB Langevin workspace wrapping `engine_diffeq`.
// [cite:engine_diffeq]

use crate::error::WasmSdkError;
use engine_diffeq::{
    BaoabLangevinSolver, BaoabParams, EulerMaruyamaSolver, LangevinSystem, OwnedWorkspace,
    OxidizeRng, SdeSystem, StochasticCalculus,
};
use math_commons::math_kernel::types::flatten_2d_index;
use wasm_bindgen::prelude::*;

/// Maximum state or Wiener dimension supported by `WasmSdeWorkspace`.
pub const MAX_SDE_DIM: usize = 4_096;

struct DiagonalOuSystem {
    dim: usize,
    theta: f64,
    mu: f64,
    sigma: f64,
}

impl SdeSystem<f64> for DiagonalOuSystem {
    fn state_dim(&self) -> usize {
        self.dim
    }

    fn wiener_dim(&self) -> usize {
        self.dim
    }

    fn drift(&self, _t: f64, x: &[f64], out_drift: &mut [f64]) {
        for (y, &xi) in out_drift.iter_mut().zip(x.iter()).take(self.dim) {
            *y = self.theta * (self.mu - xi);
        }
    }

    fn diffusion(&self, _t: f64, _x: &[f64], out_diffusion: &mut [f64]) {
        let n = self.dim;
        out_diffusion[..n * n].fill(0.0);
        for i in 0..n {
            out_diffusion[flatten_2d_index(i, i, n)] = self.sigma;
        }
    }
}

struct GeometricBrownianSystem {
    dim: usize,
    mu: f64,
    sigma: f64,
}

impl SdeSystem<f64> for GeometricBrownianSystem {
    fn state_dim(&self) -> usize {
        self.dim
    }

    fn wiener_dim(&self) -> usize {
        self.dim
    }

    fn drift(&self, _t: f64, x: &[f64], out_drift: &mut [f64]) {
        for (y, &xi) in out_drift.iter_mut().zip(x.iter()).take(self.dim) {
            *y = self.mu * xi;
        }
    }

    fn diffusion(&self, _t: f64, x: &[f64], out_diffusion: &mut [f64]) {
        let n = self.dim;
        out_diffusion[..n * n].fill(0.0);
        for (i, &xi) in x.iter().enumerate().take(n) {
            out_diffusion[flatten_2d_index(i, i, n)] = self.sigma * xi;
        }
    }
}

struct HarmonicPotential {
    dim: usize,
    omega_sq: f64,
}

impl LangevinSystem<f64> for HarmonicPotential {
    fn dim(&self) -> usize {
        self.dim
    }

    fn grad_potential(&self, q: &[f64], out_grad: &mut [f64]) {
        for (g, &qi) in out_grad.iter_mut().zip(q.iter()).take(self.dim) {
            *g = self.omega_sq * qi;
        }
    }
}

/// Reusable zero-allocation stochastic differential equation and BAOAB Langevin workspace.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WasmSdeWorkspace {
    state_dim: usize,
    wiener_dim: usize,
    rng: OxidizeRng,
    scratch: OwnedWorkspace<f64>,
    state_buf: Vec<f64>,
    momentum_buf: Vec<f64>,
}

#[wasm_bindgen]
impl WasmSdeWorkspace {
    /// Pre-allocates an SDE & Langevin workspace for `(state_dim, wiener_dim)` seeded deterministically.
    #[wasm_bindgen(js_name = tryNew)]
    pub fn try_new(state_dim: usize, wiener_dim: usize, seed: u64) -> Result<Self, WasmSdkError> {
        if state_dim == 0 || wiener_dim == 0 || state_dim > MAX_SDE_DIM || wiener_dim > MAX_SDE_DIM
        {
            return Err(WasmSdkError::invalid_argument(
                "E_SDE_DIM",
                format!(
                    "SDE dimensions ({state_dim}, {wiener_dim}) must be within 1..={MAX_SDE_DIM}"
                ),
                MAX_SDE_DIM,
                state_dim.max(wiener_dim),
            ));
        }
        let strat = EulerMaruyamaSolver::new(StochasticCalculus::Stratonovich);
        let sde_cap = strat.scratch_len(state_dim, wiener_dim);
        let baoab_cap = BaoabLangevinSolver::<f64>::scratch_len(state_dim);
        let scratch = OwnedWorkspace::try_with_capacity(sde_cap.max(baoab_cap))?;
        Ok(Self {
            state_dim,
            wiener_dim,
            rng: OxidizeRng::from_seed(seed),
            scratch,
            state_buf: vec![0.0; state_dim],
            momentum_buf: vec![0.0; state_dim],
        })
    }

    /// Reseeds the deterministic `OxidizeRng` stream in place without allocating.
    pub fn reseed(&mut self, seed: u64) {
        self.rng = OxidizeRng::from_seed(seed);
    }

    /// Configured state vector dimension.
    #[wasm_bindgen(js_name = stateDim)]
    #[must_use]
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// Byte offset of the internal `state_dim` position/state buffer in WASM linear memory.
    #[wasm_bindgen(js_name = statePtr)]
    #[must_use]
    pub fn state_ptr(&self) -> usize {
        self.state_buf.as_ptr() as usize
    }

    /// Byte offset of the internal `state_dim` momentum buffer in WASM linear memory.
    #[wasm_bindgen(js_name = momentumPtr)]
    #[must_use]
    pub fn momentum_ptr(&self) -> usize {
        self.momentum_buf.as_ptr() as usize
    }

    /// Advances an Itô Ornstein-Uhlenbeck process $dX = \theta(\mu - X)dt + \sigma dW$ in place on `state`.
    #[wasm_bindgen(js_name = stepOu)]
    pub fn step_ou(
        &mut self,
        t: f64,
        dt: f64,
        theta: f64,
        mu: f64,
        sigma: f64,
        state: &mut [f64],
    ) -> Result<(), WasmSdkError> {
        let d = state.len();
        if d == 0 || d > self.state_dim || d > self.wiener_dim {
            return Err(WasmSdkError::invalid_argument(
                "E_SDE_STATE_LEN",
                "state slice length must be within 1..=min(state_dim, wiener_dim)",
                self.state_dim.min(self.wiener_dim),
                d,
            ));
        }
        let sys = DiagonalOuSystem {
            dim: d,
            theta,
            mu,
            sigma,
        };
        let solver = EulerMaruyamaSolver::new(StochasticCalculus::Ito);
        let mut ws = self.scratch.as_workspace();
        solver.try_step(&sys, t, dt, state, &mut ws, &mut self.rng)?;
        self.state_buf[..d].copy_from_slice(state);
        Ok(())
    }

    /// Advances geometric Brownian motion $dS = \mu S\,dt + \sigma S\,dW$ (Itô or Stratonovich) in place.
    #[wasm_bindgen(js_name = stepGeometricBrownian)]
    pub fn step_geometric_brownian(
        &mut self,
        t: f64,
        dt: f64,
        mu: f64,
        sigma: f64,
        stratonovich: bool,
        state: &mut [f64],
    ) -> Result<(), WasmSdkError> {
        let d = state.len();
        if d == 0 || d > self.state_dim || d > self.wiener_dim {
            return Err(WasmSdkError::invalid_argument(
                "E_SDE_STATE_LEN",
                "state slice length must be within 1..=min(state_dim, wiener_dim)",
                self.state_dim.min(self.wiener_dim),
                d,
            ));
        }
        let calc = if stratonovich {
            StochasticCalculus::Stratonovich
        } else {
            StochasticCalculus::Ito
        };
        let sys = GeometricBrownianSystem { dim: d, mu, sigma };
        let solver = EulerMaruyamaSolver::new(calc);
        let mut ws = self.scratch.as_workspace();
        solver.try_step(&sys, t, dt, state, &mut ws, &mut self.rng)?;
        self.state_buf[..d].copy_from_slice(state);
        Ok(())
    }

    /// Advances underdamped BAOAB Langevin dynamics in a harmonic potential $V(q) = \frac{1}{2}\omega^2 \|q\|^2$.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = stepBaoabHarmonic)]
    pub fn step_baoab_harmonic(
        &mut self,
        mass: f64,
        friction: f64,
        temperature: f64,
        dt: f64,
        omega_sq: f64,
        steps: usize,
        q: &mut [f64],
        p: &mut [f64],
    ) -> Result<(), WasmSdkError> {
        let d = q.len();
        if d == 0 || d > self.state_dim || p.len() != d {
            return Err(WasmSdkError::invalid_argument(
                "E_BAOAB_DIM",
                "q and p slices must have matching length within 1..=state_dim",
                self.state_dim,
                p.len(),
            ));
        }
        let solver = BaoabLangevinSolver::try_new(BaoabParams {
            mass,
            friction,
            temperature,
            dt,
        })?;
        let sys = HarmonicPotential { dim: d, omega_sq };
        for _ in 0..steps.max(1) {
            let mut ws = self.scratch.as_workspace();
            solver.try_step(&sys, q, p, &mut ws, &mut self.rng)?;
        }
        self.state_buf[..d].copy_from_slice(q);
        self.momentum_buf[..d].copy_from_slice(p);
        Ok(())
    }
}
