//! Core zero-allocation Brent 1D parabolic + golden-section scalar minimizer.

use super::super::roots::AnalysisError;

#[derive(Clone, Copy)]
struct BrentState {
    a: f64,
    b: f64,
    x: f64,
    w: f64,
    v: f64,
    fx: f64,
    fw: f64,
    fv: f64,
    d: f64,
    e: f64,
}

fn brent_parabolic_step(st: &BrentState, tol2: f64) -> Option<f64> {
    let r = (st.x - st.w) * (st.fx - st.fv);
    let mut q = (st.x - st.v) * (st.fx - st.fw);
    let mut p = (st.x - st.v) * q - (st.x - st.w) * r;
    q = 2.0 * (q - r);
    if q > 0.0 {
        p = -p;
    } else {
        q = -q;
    }
    if p.abs() >= (0.5 * q * st.e).abs()
        || p <= q * (st.a - st.x)
        || p >= q * (st.b - st.x)
        || q == 0.0
    {
        return None;
    }
    let mut d = p / q;
    let u = st.x + d;
    if (u - st.a) < tol2 || (st.b - u) < tol2 {
        d = if st.x < 0.5 * (st.a + st.b) {
            0.5 * tol2
        } else {
            -0.5 * tol2
        };
    }
    Some(d)
}

fn brent_accept_eval(st: &mut BrentState, u: f64, fu: f64) {
    if fu <= st.fx {
        if u >= st.x {
            st.a = st.x;
        } else {
            st.b = st.x;
        }
        st.v = st.w;
        st.fv = st.fw;
        st.w = st.x;
        st.fw = st.fx;
        st.x = u;
        st.fx = fu;
    } else {
        if u < st.x {
            st.a = u;
        } else {
            st.b = u;
        }
        if fu <= st.fw || st.w == st.x {
            st.v = st.w;
            st.fv = st.fw;
            st.w = u;
            st.fw = fu;
        } else if fu <= st.fv || st.v == st.x || st.v == st.w {
            st.v = u;
            st.fv = fu;
        }
    }
}

#[verified_engine::verified]
pub(in crate::pure_math::analysis) fn brent_minimize_core<F: FnMut(f64) -> f64>(
    mut f: F,
    a: f64,
    b: f64,
    max_iterations: usize,
    tol: f64,
) -> Result<(f64, f64), AnalysisError> {
    const GOLDEN: f64 = 0.381_966_011_250_105_1;
    const ZEPS: f64 = 1e-11;
    let x0 = a + GOLDEN * (b - a);
    let fx0 = f(x0);
    if !fx0.is_finite() {
        return Err(AnalysisError::ConvergenceError(x0));
    }
    let mut st = BrentState {
        a,
        b,
        x: x0,
        w: x0,
        v: x0,
        fx: fx0,
        fw: fx0,
        fv: fx0,
        d: 0.0,
        e: 0.0,
    };
    for _ in 0..max_iterations {
        let xm = 0.5 * (st.a + st.b);
        let tol1 = tol * st.x.abs() + ZEPS;
        let tol2 = 2.0 * tol1;
        if (st.x - xm).abs() <= tol2 - 0.5 * (st.b - st.a) {
            return Ok((st.x, st.fx));
        }
        let par = if st.e.abs() > tol1 {
            brent_parabolic_step(&st, tol2)
        } else {
            None
        };
        if let Some(step) = par {
            st.e = st.d;
            st.d = step;
        } else {
            st.e = if st.x >= xm { st.a - st.x } else { st.b - st.x };
            st.d = GOLDEN * st.e;
        }
        let u = if st.d.abs() >= tol1 {
            st.x + st.d
        } else {
            st.x + st.d.signum() * tol1
        };
        let fu = f(u);
        if !fu.is_finite() {
            return Err(AnalysisError::ConvergenceError(st.x));
        }
        brent_accept_eval(&mut st, u, fu);
    }
    Err(AnalysisError::ConvergenceError(st.x))
}
