#![allow(missing_docs)]
use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::pure_math::tensor::{RiemannianMetric, christoffel_symbols};

fn main() {
    // Metric for a 2D sphere (Radius = 1.0)
    let metric = RiemannianMetric::new(|p: &DVector<f64>| {
        let theta = p[0];
        let g11 = 1.0;
        let g22 = theta.sin().powi(2);
        DMatrix::from_vec(2, 2, vec![g11, 0.0, 0.0, g22])
    });

    // Compute symbols at theta = 45 degrees
    let point = DVector::from_vec(vec![std::f64::consts::FRAC_PI_4, 0.0]);
    let gammas = christoffel_symbols(&metric, &point).expect("Singular metric");

    println!("Gamma^theta_phi_phi: {:.4}", gammas[0][(1, 1)]); // -0.5000
}
