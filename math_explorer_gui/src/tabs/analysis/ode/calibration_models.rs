#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Stack-allocated `ArrayState` ODE adapters and RK4 single-shooting helpers for [`super`].

use crate::framework::math_input::MAX_SYNTHESIZED_PARAMS;
use math_commons::ast::CompiledExpr;
use math_explorer::pure_math::analysis::ode::{
    ArrayState, OdeSystem, RungeKutta4, Solver, VectorOperations,
};
use math_explorer::pure_math::analysis::optimization::{
    CalibrationSummary, LevenbergMarquardtSolver, LevenbergMarquardtWorkspace,
    MAX_SUBSTEPS_PER_INTERVAL, ParameterBounds, Rk4ShootingObjective,
};

pub(super) struct ScalarExpOde {
    pub(super) k: f64,
}

impl OdeSystem<ArrayState<1>> for ScalarExpOde {
    fn derivative(&self, _t: f64, state: &ArrayState<1>) -> ArrayState<1> {
        ArrayState([self.k * state.0[0]])
    }
}

pub(super) struct ScalarLogisticOde {
    pub(super) r: f64,
    pub(super) cap_k: f64,
}

impl OdeSystem<ArrayState<1>> for ScalarLogisticOde {
    fn derivative(&self, _t: f64, state: &ArrayState<1>) -> ArrayState<1> {
        let y = state.0[0];
        ArrayState([self.r * y * (1.0 - y / self.cap_k)])
    }
}

pub(super) struct CompiledScalarOde {
    pub(super) expr: CompiledExpr,
    pub(super) params: [f64; MAX_SYNTHESIZED_PARAMS],
    pub(super) param_len: usize,
}

impl OdeSystem<ArrayState<1>> for CompiledScalarOde {
    fn derivative(&self, t: f64, state: &ArrayState<1>) -> ArrayState<1> {
        ArrayState([self
            .expr
            .eval_2d(t, state.0[0], &self.params[..self.param_len])])
    }
}

pub(super) struct CompiledCoupledOde {
    pub(super) dy: CompiledExpr,
    pub(super) dv: CompiledExpr,
    pub(super) params: [f64; MAX_SYNTHESIZED_PARAMS],
    pub(super) param_len: usize,
}

impl OdeSystem<ArrayState<2>> for CompiledCoupledOde {
    fn derivative(&self, t: f64, state: &ArrayState<2>) -> ArrayState<2> {
        let p = &self.params[..self.param_len];
        ArrayState([
            self.dy.eval_3d(t, state.0[0], state.0[1], p),
            self.dv.eval_3d(t, state.0[0], state.0[1], p),
        ])
    }
}

pub(super) struct OscArrayOde {
    pub(super) k: f64,
}

impl OdeSystem<ArrayState<2>> for OscArrayOde {
    fn derivative(&self, _t: f64, state: &ArrayState<2>) -> ArrayState<2> {
        ArrayState([state.0[1], -self.k * state.0[0]])
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn solve_shooting<Sys, St, C, O, F>(
    system: Sys,
    example: &St,
    times: &[f64],
    observed: &[f64],
    max_dt: f64,
    init: &[f64],
    bounds: &[ParameterBounds],
    configure: C,
    observe: O,
    is_finite: F,
) -> Result<CalibrationSummary, String>
where
    Sys: OdeSystem<St>,
    St: VectorOperations,
    C: FnMut(&mut Sys, &[f64], f64, &mut St) -> bool,
    O: Fn(&St) -> f64,
    F: Fn(&St) -> bool,
{
    let mut obj = Rk4ShootingObjective::new(
        system, example, times, observed, max_dt, configure, observe, is_finite,
    )
    .map_err(|e| e.to_string())?;
    let mut ws =
        LevenbergMarquardtWorkspace::new(times.len(), init.len()).map_err(|e| e.to_string())?;
    LevenbergMarquardtSolver::default()
        .fit(&mut obj, init, bounds, &mut ws)
        .map_err(|e| e.to_string())
}

pub(super) fn advance_from_zero<S, St, F>(
    sys: &S,
    t0: f64,
    max_dt: f64,
    state: &mut St,
    is_finite: F,
) -> bool
where
    S: OdeSystem<St>,
    St: VectorOperations,
    F: Fn(&St) -> bool,
{
    if t0 <= 0.0 {
        return is_finite(state);
    }
    let steps = ((t0 / max_dt).ceil() as usize).clamp(1, MAX_SUBSTEPS_PER_INTERVAL);
    let dt = t0 / (steps as f64);
    let mut rk4 = RungeKutta4::new(state);
    let mut t = 0.0;
    for _ in 0..steps {
        rk4.step(sys, t, state, dt);
        if !is_finite(state) {
            return false;
        }
        t += dt;
    }
    true
}
