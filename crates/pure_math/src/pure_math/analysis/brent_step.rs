// [cite:advanced_linear_algebra]

#[derive(Clone, Copy)]
pub(super) struct BrentIterate {
    pub(super) a: f64,
    pub(super) b: f64,
    pub(super) c: f64,
    pub(super) fa: f64,
    pub(super) fb: f64,
    pub(super) fc: f64,
    pub(super) d: f64,
    pub(super) e: f64,
}

#[verified_engine::verified]
pub(super) fn brent_propose_step(st: &BrentIterate, xm: f64, tol1: f64) -> (f64, f64) {
    if st.e.abs() < tol1 || st.fa.abs() <= st.fb.abs() {
        return (xm, xm);
    }
    let s = st.fb / st.fa;
    let (mut p, mut q) = if (st.a - st.c).abs() <= f64::EPSILON {
        (2.0 * xm * s, 1.0 - s)
    } else {
        let qi = st.fa / st.fc;
        let r = st.fb / st.fc;
        (
            s * (2.0 * xm * qi * (qi - r) - (st.b - st.a) * (r - 1.0)),
            (qi - 1.0) * (r - 1.0) * (s - 1.0),
        )
    };
    if p > 0.0 {
        q = -q;
    }
    p = p.abs();
    let bound_bracket = 3.0 * xm * q - (tol1 * q).abs();
    let bound_prev = (st.e * q).abs();
    if q != 0.0 && 2.0 * p < bound_bracket.min(bound_prev) {
        (p / q, st.d)
    } else {
        (xm, xm)
    }
}
