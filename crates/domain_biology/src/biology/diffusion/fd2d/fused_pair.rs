use crate::biology::morphogenesis::reaction::ReactionKinetics;
use pure_math::pure_math::analysis::pde::fused_stepper::StencilOperators;

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn eval_pair_cell<const N: usize, K: ReactionKinetics<N>>(
    uc: f64,
    ul: f64,
    ur: f64,
    uu: f64,
    ud: f64,
    vc: f64,
    vl: f64,
    vr: f64,
    vu: f64,
    vd: f64,
    dt: f64,
    coeffs: [f64; 2],
    ops: &StencilOperators,
    kinetics: &K,
) -> (f64, f64) {
    let mut curr = [0.0; N];
    curr[0] = uc;
    curr[1] = vc;
    let rates = kinetics.reaction(curr);
    let d2u = ops.central_diff_2nd_2d(uc, ul, ur, uu, ud);
    let d2v = ops.central_diff_2nd_2d(vc, vl, vr, vu, vd);
    (
        uc + dt * (coeffs[0] * d2u + rates[0]),
        vc + dt * (coeffs[1] * d2v + rates[1]),
    )
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub(super) fn step_fused_pair<const N: usize, K: ReactionKinetics<N>>(
    dim: (usize, usize),
    u: &[f64],
    v: &[f64],
    nu: &mut [f64],
    nv: &mut [f64],
    dt: f64,
    coeffs: [f64; 2],
    ops: &StencilOperators,
    kinetics: &K,
) {
    let (w, h) = dim;
    let n = w.saturating_mul(h);
    if n == 0 || u.len() < n || v.len() < n || nu.len() < n || nv.len() < n {
        return;
    }
    for y in 0..h {
        let off = y * w;
        let uoff = if y > 0 { off - w } else { off };
        let doff = if y + 1 < h { off + w } else { off };
        let (u_up, u_mid, u_dn) = (&u[uoff..uoff + w], &u[off..off + w], &u[doff..doff + w]);
        let (v_up, v_mid, v_dn) = (&v[uoff..uoff + w], &v[off..off + w], &v[doff..doff + w]);
        let (nu_r, nv_r) = (&mut nu[off..off + w], &mut nv[off..off + w]);
        let r0 = usize::from(w > 1);
        (nu_r[0], nv_r[0]) = eval_pair_cell(
            u_mid[0], u_mid[0], u_mid[r0], u_up[0], u_dn[0], v_mid[0], v_mid[0], v_mid[r0],
            v_up[0], v_dn[0], dt, coeffs, ops, kinetics,
        );
        if w > 2 {
            for (((((((&uu, um), &ud), &vu), vm), &vd), out_u), out_v) in u_up[1..w - 1]
                .iter()
                .zip(u_mid.windows(3))
                .zip(&u_dn[1..w - 1])
                .zip(&v_up[1..w - 1])
                .zip(v_mid.windows(3))
                .zip(&v_dn[1..w - 1])
                .zip(&mut nu_r[1..w - 1])
                .zip(&mut nv_r[1..w - 1])
            {
                (*out_u, *out_v) = eval_pair_cell(
                    um[1], um[0], um[2], uu, ud, vm[1], vm[0], vm[2], vu, vd, dt, coeffs, ops,
                    kinetics,
                );
            }
        }
        if w > 1 {
            let l = w - 1;
            (nu_r[l], nv_r[l]) = eval_pair_cell(
                u_mid[l],
                u_mid[l - 1],
                u_mid[l],
                u_up[l],
                u_dn[l],
                v_mid[l],
                v_mid[l - 1],
                v_mid[l],
                v_up[l],
                v_dn[l],
                dt,
                coeffs,
                ops,
                kinetics,
            );
        }
    }
}
