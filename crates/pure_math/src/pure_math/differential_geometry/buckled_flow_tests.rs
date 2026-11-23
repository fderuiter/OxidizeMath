#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::pure_math::differential_geometry::surface::{KleinBottle, Sphere, Torus};
use std::f64::consts::{PI, TAU};
use std::sync::atomic::Ordering;
use verified_engine::allocator::MEMORY_LOCK;
use verified_engine::engine::VerifiedEngine;

#[test]
fn buckled_flow_executes_with_zero_heap_allocations_in_verified_engine() {
    let torus = Torus {
        major_radius: 2.0,
        minor_radius: 0.65,
    };
    let mut flow = BuckledSurfaceFlow::from_parametric(
        &torus,
        SurfaceTopology::PeriodicTorus,
        (0.0, TAU),
        (0.0, TAU),
        16,
        16,
    );
    let morphogen: Vec<f64> = (0..(16 * 16))
        .map(|idx| 1.0 + 0.35 * (idx as f64 * 0.25).sin())
        .collect();

    let (ok, metrics) =
        VerifiedEngine::run_verified(|| flow.step_buckled(&morphogen, 1.0, 0.25, 0.08, 0.1, 0.005));
    assert!(ok);
    assert!(metrics.function_calls >= 1);
    assert!(!MEMORY_LOCK.load(Ordering::SeqCst));
    assert!(flow.displacements().iter().any(|&d| d.abs() > 1e-5));
}

#[test]
fn buckled_flow_pushes_morphogen_peaks_outward_and_rolls_back_non_finite() {
    let sphere = Sphere { radius: 1.5 };
    let (nu, nv) = (16, 13);
    let mut flow = BuckledSurfaceFlow::from_parametric(
        &sphere,
        SurfaceTopology::PolarSphere,
        (0.0, TAU),
        (0.0, PI),
        nu,
        nv,
    );
    let mut morphogen = vec![0.5_f64; nu * nv];
    let peak_idx = flatten_2d_index(nv / 2, 0, nv);
    morphogen[peak_idx] = 2.0;

    for _ in 0..8 {
        assert!(flow.step_buckled(&morphogen, 0.5, 0.3, 0.05, 0.05, 0.005));
    }
    assert!(
        flow.points()[peak_idx].coords.norm() > 1.5,
        "Positive morphogen peak on Sphere must buckle outward (> R)"
    );
    // North pole vertices remain welded to a single shared coordinate
    let north_0 = flow.points()[flatten_2d_index(0, 0, nv)];
    for i in 1..nu {
        let north_i = flow.points()[flatten_2d_index(0, i, nv)];
        assert!((north_0 - north_i).norm() < 1e-12);
    }

    let saved = flow.points().to_vec();
    let mut bad = morphogen;
    bad[peak_idx] = f64::NAN;
    assert!(!flow.step_buckled(&bad, 0.5, 0.3, 0.05, 0.05, 0.005));
    assert_eq!(flow.points(), saved.as_slice());
}

struct ReversingFigureEightKlein {
    radius: f64,
}

impl ParametricSurface for ReversingFigureEightKlein {
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        let half_u = 0.5 * u;
        let a = self.radius + half_u.cos() * v.sin() - half_u.sin() * (2.0 * v).sin();
        Point3::new(
            a * u.cos(),
            a * u.sin(),
            half_u.sin() * v.sin() + half_u.cos() * (2.0 * v).sin(),
        )
    }
}

#[test]
fn klein_bottle_twisted_seam_bijection_matches_both_immersions() {
    let klein = KleinBottle { radius: 2.0 };
    let (fwd, rev) = detect_twisted_seam_map(&klein, (0.0, TAU), (0.0, TAU), 16);
    assert_eq!(fwd[0], 8);
    assert_eq!(fwd[3], 11);
    assert_eq!(rev[11], 3);

    let rev_klein = ReversingFigureEightKlein { radius: 3.0 };
    let (rfwd, rrev) = detect_twisted_seam_map(&rev_klein, (0.0, TAU), (0.0, TAU), 16);
    assert_eq!(rfwd[0], 0);
    assert_eq!(rfwd[4], 12);
    assert_eq!(rrev[12], 4);
}
