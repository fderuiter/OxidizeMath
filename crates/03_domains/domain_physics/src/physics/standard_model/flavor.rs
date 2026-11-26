//! Flavor Physics deals with quark mixing via the CKM matrix.

use nalgebra::Matrix3;
use num_complex::Complex;

/// Constructs the Cabibbo-Kobayashi-Maskawa (CKM) Matrix.
///
/// The CKM matrix describes the mixing between mass eigenstates and weak interaction eigenstates of quarks.
/// It is parametrized by three mixing angles ($\theta_{12}, \theta_{23}, \theta_{13}$) and one CP-violating phase ($\delta$).
///
/// # Arguments
/// * `theta12`: Mixing angle $\theta_{12}$ (radians).
/// * `theta23`: Mixing angle $\theta_{23}$ (radians).
/// * `theta13`: Mixing angle $\theta_{13}$ (radians).
/// * `delta`: CP-violating phase $\delta$ (radians).
///
/// # Returns
/// A 3x3 Complex Matrix representing $V_{CKM}$.
#[verified_engine::verified]
pub fn construct_ckm(
    theta12: f64,
    theta23: f64,
    theta13: f64,
    delta: f64,
) -> Matrix3<Complex<f64>> {
    let c12 = theta12.cos();
    let s12 = theta12.sin();
    let c23 = theta23.cos();
    let s23 = theta23.sin();
    let c13 = theta13.cos();
    let s13 = theta13.sin();

    let phase_pos = Complex::from_polar(1.0, delta);
    let phase_neg = Complex::from_polar(1.0, -delta);

    // Row 1
    let v_ud = Complex::new(c12 * c13, 0.0);
    let v_us = Complex::new(s12 * c13, 0.0);
    let v_ub = Complex::new(s13, 0.0) * phase_neg;

    // Row 2
    // -s12 c23 - c12 s23 s13 e^{i delta}
    let v_cd = Complex::new(-s12 * c23, 0.0) - Complex::new(c12 * s23 * s13, 0.0) * phase_pos;
    // c12 c23 - s12 s23 s13 e^{i delta}
    let v_cs = Complex::new(c12 * c23, 0.0) - Complex::new(s12 * s23 * s13, 0.0) * phase_pos;
    let v_cb = Complex::new(s23 * c13, 0.0);

    // Row 3
    // s12 s23 - c12 c23 s13 e^{i delta}
    let v_td = Complex::new(s12 * s23, 0.0) - Complex::new(c12 * c23 * s13, 0.0) * phase_pos;
    // -c12 s23 - s12 c23 s13 e^{i delta}
    let v_ts = Complex::new(-c12 * s23, 0.0) - Complex::new(s12 * c23 * s13, 0.0) * phase_pos;
    let v_tb = Complex::new(c23 * c13, 0.0);

    Matrix3::new(v_ud, v_us, v_ub, v_cd, v_cs, v_cb, v_td, v_ts, v_tb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_ckm_unitarity() {
        let theta12 = 0.2;
        let theta23 = 0.04;
        let theta13 = 0.003;
        let delta = 1.2;

        let v_ckm = construct_ckm(theta12, theta23, theta13, delta);
        let v_dag = v_ckm.adjoint();
        let identity = v_ckm * v_dag;

        for i in 0..3 {
            for j in 0..3 {
                let val = identity[(i, j)];
                if i == j {
                    assert_relative_eq!(
                        val.re,
                        1.0,
                        epsilon = math_commons::registry::TOLERANCE_HIGH
                    );
                    assert_relative_eq!(
                        val.im,
                        0.0,
                        epsilon = math_commons::registry::TOLERANCE_HIGH
                    );
                } else {
                    assert_relative_eq!(
                        val.re,
                        0.0,
                        epsilon = math_commons::registry::TOLERANCE_HIGH
                    );
                    assert_relative_eq!(
                        val.im,
                        0.0,
                        epsilon = math_commons::registry::TOLERANCE_HIGH
                    );
                }
            }
        }
    }
}
