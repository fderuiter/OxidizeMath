#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Stack-only zero-allocation ODE dynamics and 2x2 Jacobian regime classifier for [`super::OdeRegimeCartographer`].

use super::OdeAtlasRegime;
use crate::async_sim::atlas::AtlasCell;
use crate::framework::math_input::MAX_SYNTHESIZED_PARAMS;
use math_commons::ast::CompiledExpr;
use math_explorer::pure_math::analysis::ode::{
    ArrayState, Euler, GaussLegendre4, Heun, ImplicitMidpoint, IntegrationMethod, OdeModel,
    OdeSystem, RungeKutta4, Solver, TimeStepper,
};

const ATLAS_EVAL_STEPS: usize = 80;

#[derive(Clone, Copy)]
pub(super) enum StackOdeDynamics<'a> {
    Exponential {
        k: f64,
    },
    DampedHarmonic {
        k: f64,
        c: f64,
    },
    Logistic {
        r: f64,
        cap_k: f64,
    },
    Linear2D {
        matrix_a: [[f64; 2]; 2],
    },
    ScalarExpr {
        expr: &'a CompiledExpr,
        params: [f64; MAX_SYNTHESIZED_PARAMS],
        param_len: usize,
    },
    CoupledExpr {
        dy: &'a CompiledExpr,
        dv: &'a CompiledExpr,
        params: [f64; MAX_SYNTHESIZED_PARAMS],
        param_len: usize,
    },
}

impl StackOdeDynamics<'_> {
    #[inline]
    pub(super) const fn is_2d(&self) -> bool {
        matches!(
            self,
            Self::DampedHarmonic { .. } | Self::Linear2D { .. } | Self::CoupledExpr { .. }
        )
    }
}

impl OdeSystem<ArrayState<2>> for StackOdeDynamics<'_> {
    fn derivative(&self, t: f64, state: &ArrayState<2>) -> ArrayState<2> {
        let mut out = ArrayState::zeros();
        self.derivative_in_place(t, state, &mut out);
        out
    }

    #[inline]
    fn derivative_in_place(&self, t: f64, state: &ArrayState<2>, out: &mut ArrayState<2>) {
        let y = state.0[0];
        let v = state.0[1];
        match self {
            Self::Exponential { k } => {
                out.0[0] = *k * y;
                out.0[1] = 0.0;
            }
            Self::DampedHarmonic { k, c } => {
                out.0[0] = v;
                out.0[1] = -(*k) * y - (*c) * v;
            }
            Self::Logistic { r, cap_k } => {
                out.0[0] = if cap_k.abs() > 1e-12 {
                    *r * y * (1.0 - y / *cap_k)
                } else {
                    f64::NAN
                };
                out.0[1] = 0.0;
            }
            Self::Linear2D { matrix_a } => {
                out.0[0] = matrix_a[0][0] * y + matrix_a[0][1] * v;
                out.0[1] = matrix_a[1][0] * y + matrix_a[1][1] * v;
            }
            Self::ScalarExpr {
                expr,
                params,
                param_len,
            } => {
                out.0[0] = expr.eval_2d(t, y, &params[..*param_len]);
                out.0[1] = 0.0;
            }
            Self::CoupledExpr {
                dy,
                dv,
                params,
                param_len,
            } => {
                let p = &params[..*param_len];
                out.0[0] = dy.eval_3d(t, y, v, p);
                out.0[1] = dv.eval_3d(t, y, v, p);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct TrajectorySummary {
    finite: bool,
    r_init: f64,
    early_max_r: f64,
    late_max_r: f64,
    late_min_r: f64,
    end_state: ArrayState<2>,
    t_end: f64,
    zero_crossings: u32,
}

fn integrate_with_solver<S: Solver<ArrayState<2>>>(
    mut model: OdeModel<ArrayState<2>, StackOdeDynamics<'_>, S>,
    dt: f64,
) -> TrajectorySummary {
    let r_init = model.state.0[0].hypot(model.state.0[1]);
    let mut early_max_r = r_init;
    let mut late_max_r = 0.0_f64;
    let mut late_min_r = f64::INFINITY;
    let mut prev_sig = if model.dynamics.is_2d() {
        model.state.0[1]
    } else {
        model.state.0[0]
    };
    let mut zero_crossings = 0_u32;
    let half = ATLAS_EVAL_STEPS / 2;

    for step in 0..ATLAS_EVAL_STEPS {
        model.step(dt);
        let y = model.state.0[0];
        let v = model.state.0[1];
        if !y.is_finite() || !v.is_finite() || y.abs() > 1e6 || v.abs() > 1e6 {
            return TrajectorySummary {
                finite: false,
                r_init,
                early_max_r,
                late_max_r: 1e6,
                late_min_r: 0.0,
                end_state: model.state,
                t_end: (step + 1) as f64 * dt,
                zero_crossings,
            };
        }
        let r = y.hypot(v);
        if step < half {
            early_max_r = early_max_r.max(r);
        } else {
            late_max_r = late_max_r.max(r);
            late_min_r = late_min_r.min(r);
        }
        let sig = if model.dynamics.is_2d() { v } else { y };
        if (prev_sig > 1e-6 && sig < -1e-6) || (prev_sig < -1e-6 && sig > 1e-6) {
            zero_crossings += 1;
        }
        if sig.abs() > 1e-6 {
            prev_sig = sig;
        }
    }

    TrajectorySummary {
        finite: true,
        r_init,
        early_max_r,
        late_max_r,
        late_min_r: if late_min_r.is_finite() {
            late_min_r
        } else {
            late_max_r
        },
        end_state: model.state,
        t_end: ATLAS_EVAL_STEPS as f64 * dt,
        zero_crossings,
    }
}

fn jacobian_invariants_2d(
    dyn_sys: &StackOdeDynamics<'_>,
    t: f64,
    center: ArrayState<2>,
) -> (f64, f64, f64) {
    let eps = 1e-5;
    let mut fp = ArrayState::zeros();
    let mut fm = ArrayState::zeros();

    dyn_sys.derivative_in_place(t, &ArrayState([center.0[0] + eps, center.0[1]]), &mut fp);
    dyn_sys.derivative_in_place(t, &ArrayState([center.0[0] - eps, center.0[1]]), &mut fm);
    let j00 = (fp.0[0] - fm.0[0]) / (2.0 * eps);
    let j10 = (fp.0[1] - fm.0[1]) / (2.0 * eps);

    dyn_sys.derivative_in_place(t, &ArrayState([center.0[0], center.0[1] + eps]), &mut fp);
    dyn_sys.derivative_in_place(t, &ArrayState([center.0[0], center.0[1] - eps]), &mut fm);
    let j01 = (fp.0[0] - fm.0[0]) / (2.0 * eps);
    let j11 = (fp.0[1] - fm.0[1]) / (2.0 * eps);

    let trace = j00 + j11;
    let det = j00 * j11 - j01 * j10;
    let disc = trace * trace - 4.0 * det;
    (trace, det, disc)
}

pub(super) fn classify_stack_dynamics(
    dyn_sys: StackOdeDynamics<'_>,
    method: IntegrationMethod,
    init: ArrayState<2>,
    dt: f64,
) -> AtlasCell {
    if !dt.is_finite() || dt <= 0.0 || !init.is_finite() {
        return AtlasCell {
            regime: OdeAtlasRegime::UnstableDivergence as u8,
            primary: 1.0,
            secondary: f64::NAN,
        };
    }
    let seeded = if init.0[0].hypot(init.0[1]) < 1e-5 {
        ArrayState([1.0, 0.0])
    } else {
        init
    };
    let summary = match method {
        IntegrationMethod::RungeKutta4 => integrate_with_solver(
            OdeModel::new(seeded, dyn_sys, RungeKutta4::new(&seeded)),
            dt,
        ),
        IntegrationMethod::Euler => {
            integrate_with_solver(OdeModel::new(seeded, dyn_sys, Euler::new(&seeded)), dt)
        }
        IntegrationMethod::Heun => {
            integrate_with_solver(OdeModel::new(seeded, dyn_sys, Heun::new(&seeded)), dt)
        }
        IntegrationMethod::ImplicitMidpoint => integrate_with_solver(
            OdeModel::new(seeded, dyn_sys, ImplicitMidpoint::new(&seeded)),
            dt,
        ),
        IntegrationMethod::GaussLegendre4 => integrate_with_solver(
            OdeModel::new(seeded, dyn_sys, GaussLegendre4::new(&seeded)),
            dt,
        ),
    };
    classify_summary(&dyn_sys, summary)
}

fn classify_summary(dyn_sys: &StackOdeDynamics<'_>, s: TrajectorySummary) -> AtlasCell {
    if !s.finite || s.late_max_r > 1e4 {
        return AtlasCell {
            regime: OdeAtlasRegime::UnstableDivergence as u8,
            primary: 1.0,
            secondary: s.late_max_r,
        };
    }
    let amp_ratio = (s.late_max_r / s.r_init.max(1e-6)).clamp(0.0, 100.0);
    if !dyn_sys.is_2d() {
        let regime = if s.late_max_r > 1.25 * s.early_max_r && s.early_max_r > 1.05 * s.r_init {
            OdeAtlasRegime::UnstableDivergence
        } else if s.zero_crossings >= 2 && (s.late_max_r - s.late_min_r) > 0.05 {
            OdeAtlasRegime::SustainedCycle
        } else {
            OdeAtlasRegime::OverdampedDecay
        };
        return AtlasCell {
            regime: regime as u8,
            primary: (amp_ratio / 2.5).clamp(0.0, 1.0),
            secondary: s.late_max_r,
        };
    }

    let mut f_origin = ArrayState::zeros();
    dyn_sys.derivative_in_place(0.0, &ArrayState([0.0, 0.0]), &mut f_origin);
    let origin_is_eq = f_origin.0[0].hypot(f_origin.0[1]) < 1e-4;
    let probe = if origin_is_eq && s.late_max_r < s.early_max_r {
        ArrayState([0.0, 0.0])
    } else {
        s.end_state
    };
    let (trace, det, disc) = jacobian_invariants_2d(dyn_sys, s.t_end, probe);
    let expanding = s.late_max_r > 1.08 * s.early_max_r && s.early_max_r > 1.02 * s.r_init;
    let decaying = s.late_max_r < 0.992 * s.early_max_r && trace < -1e-4 && det > 1e-5;
    let settled = (s.late_max_r - s.late_min_r) < 0.04 * s.r_init.max(1.0) && !expanding;

    let regime = if expanding && (s.late_max_r > 12.0 || trace > 1e-4 || det < -1e-4) {
        OdeAtlasRegime::UnstableDivergence
    } else if decaying || settled {
        if disc < -1e-7 {
            OdeAtlasRegime::UnderdampedSpiral
        } else {
            OdeAtlasRegime::OverdampedDecay
        }
    } else if s.zero_crossings >= 1 {
        OdeAtlasRegime::SustainedCycle
    } else if disc < -1e-7 {
        OdeAtlasRegime::UnderdampedSpiral
    } else {
        OdeAtlasRegime::OverdampedDecay
    };

    AtlasCell {
        regime: regime as u8,
        primary: (amp_ratio / 2.5).clamp(0.0, 1.0),
        secondary: disc,
    }
}
