use super::*;
use crate::pure_math::statistics::tda::complex::{vietoris_rips_complex, vietoris_rips_complex_3d};
use crate::pure_math::statistics::tda::core::{
    Point2D, Point3D, PointCloud, PointCloud3D, Simplex, farthest_point_landmarks,
};

#[test]
#[verified_engine::verified]
fn test_betti_number_0_single_and_disconnected() {
    let mut complex = SimplicialComplex::new();
    complex.add_simplex(Simplex::new(vec![0]).unwrap()).unwrap();
    assert_eq!(betti_number_0(&complex).unwrap(), 1);

    complex.add_simplex(Simplex::new(vec![1]).unwrap()).unwrap();
    assert_eq!(betti_number_0(&complex).unwrap(), 2);

    complex
        .add_simplex(Simplex::new(vec![0, 1]).unwrap())
        .unwrap();
    assert_eq!(betti_number_0(&complex).unwrap(), 1);
}

#[test]
#[verified_engine::verified]
fn test_betti_number_1_hollow_and_filled_triangles() {
    let mut hollow = SimplicialComplex::new();
    hollow
        .add_simplex(Simplex::new(vec![0, 1]).unwrap())
        .unwrap();
    hollow
        .add_simplex(Simplex::new(vec![1, 2]).unwrap())
        .unwrap();
    hollow
        .add_simplex(Simplex::new(vec![2, 0]).unwrap())
        .unwrap();
    assert_eq!(betti_number_1(&hollow).unwrap(), 1);

    // Adding isolated vertices makes e < v (3 < 5), yet the hollow cycle persists.
    hollow.add_simplex(Simplex::new(vec![3]).unwrap()).unwrap();
    hollow.add_simplex(Simplex::new(vec![4]).unwrap()).unwrap();
    assert_eq!(betti_number_1(&hollow).unwrap(), 1);

    let mut filled = SimplicialComplex::new();
    filled
        .add_simplex(Simplex::new(vec![0, 1, 2]).unwrap())
        .unwrap();
    assert_eq!(betti_number_1(&filled).unwrap(), 0);
}

#[test]
#[verified_engine::verified]
fn test_betti_numbers_clusters_circle_and_line() {
    let clusters = PointCloud::new(vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 0.0),
        Point2D::new(10.0, 0.0),
        Point2D::new(11.0, 0.0),
    ])
    .unwrap();
    assert_eq!(
        betti_numbers(&vietoris_rips_complex(&clusters, 1.5).unwrap()).unwrap(),
        (2, 0)
    );

    let circle = PointCloud::new(
        (0..8)
            .map(|i| {
                let a = 2.0 * std::f64::consts::PI * (i as f64) / 8.0;
                Point2D::new(a.cos(), a.sin())
            })
            .collect(),
    )
    .unwrap();
    assert_eq!(
        betti_numbers(&vietoris_rips_complex(&circle, 0.8).unwrap()).unwrap(),
        (1, 1)
    );

    let line = PointCloud::new(vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 0.0),
        Point2D::new(2.0, 0.0),
        Point2D::new(3.0, 0.0),
    ])
    .unwrap();
    assert_eq!(
        betti_numbers(&vietoris_rips_complex(&line, 1.5).unwrap()).unwrap(),
        (1, 0)
    );
}

#[test]
#[verified_engine::verified]
fn test_betti_number_1_four_clique_with_disjoint_four_cycle() {
    // 4-clique on {0,1,2,3} has 4 triangles whose boundaries sum to 0 mod 2 (rank = 3),
    // plus a disjoint hollow 4-cycle on {4,5,6,7} (1 independent cycle).
    // Exact Z2 homology gives beta_0 = 2 and beta_1 = (10 - 8 + 2) - 3 = 1.
    let mut complex = SimplicialComplex::new();
    for tri in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
        complex
            .add_simplex(Simplex::new(tri.to_vec()).unwrap())
            .unwrap();
    }
    for edge in [[4, 5], [5, 6], [6, 7], [7, 4]] {
        complex
            .add_simplex(Simplex::new(edge.to_vec()).unwrap())
            .unwrap();
    }

    assert_eq!(betti_number_0(&complex).unwrap(), 2);
    assert_eq!(betti_number_1(&complex).unwrap(), 1);
}

#[test]
#[verified_engine::verified]
fn test_betti_number_1_retains_cycle_near_attached_k4_clique() {
    let mut complex = SimplicialComplex::new();
    for &(u, v, w) in &[(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)] {
        complex
            .add_simplex(Simplex::new(vec![u, v, w]).unwrap())
            .unwrap();
    }
    for &(u, v) in &[(3, 4), (4, 5), (5, 6), (6, 3)] {
        complex
            .add_simplex(Simplex::new(vec![u, v]).unwrap())
            .unwrap();
    }
    assert_eq!(betti_numbers(&complex).unwrap(), (1, 1));
}

#[verified_engine::verified]
fn lorenz_rk4_step(s: Point3D, dt: f64, sigma: f64, rho: f64, beta: f64) -> Point3D {
    let deriv = |p: Point3D| -> Point3D {
        Point3D::new(
            sigma * (p.y - p.x),
            p.x * (rho - p.z) - p.y,
            p.x * p.y - beta * p.z,
        )
    };
    let add_scaled = |a: Point3D, b: Point3D, w: f64| -> Point3D {
        Point3D::new(a.x + w * b.x, a.y + w * b.y, a.z + w * b.z)
    };
    let k1 = deriv(s);
    let k2 = deriv(add_scaled(s, k1, 0.5 * dt));
    let k3 = deriv(add_scaled(s, k2, 0.5 * dt));
    let k4 = deriv(add_scaled(s, k3, dt));
    Point3D::new(
        s.x + (dt / 6.0) * (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x),
        s.y + (dt / 6.0) * (k1.y + 2.0 * k2.y + 2.0 * k3.y + k4.y),
        s.z + (dt / 6.0) * (k1.z + 2.0 * k2.z + 2.0 * k3.z + k4.z),
    )
}

#[test]
#[verified_engine::verified]
fn test_betti_number_1_chaotic_lorenz_landmarks_detects_two_wings() {
    let (sigma, rho, beta, dt) = (10.0, 28.0, 8.0 / 3.0, 0.01);
    let mut state = Point3D::new(1.0, 1.0, 1.0);
    for _ in 0..800 {
        state = lorenz_rk4_step(state, dt, sigma, rho, beta);
    }
    let mut trajectory = Vec::with_capacity(2400);
    for _ in 0..2400 {
        state = lorenz_rk4_step(state, dt, sigma, rho, beta);
        trajectory.push(state);
    }
    let landmarks = farthest_point_landmarks(&trajectory, 64).unwrap();
    assert_eq!(landmarks.len(), 64);
    let cloud = PointCloud3D::new(landmarks).unwrap();

    // Sweep butterfly wing scale and confirm exact Z2 homology locks onto beta_1 = 2
    // with a single connected strange-attractor component (beta_0 = 1).
    let mut found_two_loops = false;
    for step in 12..=22 {
        let radius = step as f64 * 0.5;
        let complex = vietoris_rips_complex_3d(&cloud, radius).unwrap();
        let (b0, b1) = betti_numbers(&complex).unwrap();
        if b0 == 1 && b1 == 2 {
            found_two_loops = true;
            break;
        }
    }
    assert!(
        found_two_loops,
        "Expected 64-landmark chaotic Lorenz attractor to exhibit (beta_0=1, beta_1=2)"
    );
}
