//! Cross-platform deterministic tests for `domain_biology` Lenia and Turing morphogenesis suites.

use domain_biology::biology::lenia::engine::LeniaEngine;
use domain_biology::biology::lenia::kernel::{growth_gaussian, kernel_core_bump};
use domain_biology::biology::lenia::presets::{LeniaSpecies, stamp_species_at};
use domain_biology::biology::morphogenesis::{DiffusionCoeff, TuringSystem};
use math_commons::math_kernel::types::{Dimension, StepSize};
use oxidize_core::rng::{OxidizeRng, TrajectoryDigest};
use rand::Rng;

const PINNED_LENIA_SEEDING_DIGEST: &str =
    "2055b621ff9f94bee8a45d3c1c1aba2dd8da81c2219ac05427369b513e4b9fc7";
const PINNED_TURING_DIGEST: &str =
    "1ea8208eeb8c37e8265536ed8d37be9b7c395fcceb55e9ea939ac3d53329297d";

#[test]
#[verified_engine::verified]
fn test_lenia_species_seeding_and_kernels_pinned_digest() {
    let run = |seed: u64| -> (Vec<f64>, String) {
        let mut rng = OxidizeRng::new(seed);
        let mut engine = LeniaEngine::from_species(32, 32, LeniaSpecies::Orbium, 0.05, &mut rng);
        stamp_species_at(&mut engine, LeniaSpecies::Scutium, 8, 8, 0.04, &mut rng);

        let mut digest = TrajectoryDigest::new();
        digest.update_f64_slice(engine.state());

        for k in 0..=10 {
            let r = (k as f64) * 0.1;
            digest.update_f64(kernel_core_bump(r));
            digest.update_f64(growth_gaussian(r * 0.3, 0.15, 0.015));
        }
        (engine.state().to_vec(), digest.finish_hex())
    };

    let (s1, h1) = run(2026);
    let (s2, h2) = run(2026);
    let (s3, h3) = run(2027);
    assert_eq!(s1, s2);
    assert_eq!(h1, h2);
    assert_ne!(s1, s3);
    assert_ne!(h1, h3);
    assert_eq!(h1, PINNED_LENIA_SEEDING_DIGEST);
}

#[test]
#[verified_engine::verified]
fn test_turing_morphogenesis_seeded_trajectory_pinned_digest() {
    let run = |seed: u64| -> (Vec<f64>, String) {
        let n = 32;
        let mut system = TuringSystem::new(
            Dimension(n),
            DiffusionCoeff(1.0),
            DiffusionCoeff(20.0),
            StepSize(1.0),
        );
        let mut rng = OxidizeRng::new(seed);
        for i in 0..n {
            system.u_mut()[i] = 1.0 + rng.gen_range(-0.08..0.08);
            system.v_mut()[i] = 0.5 + rng.gen_range(-0.08..0.08);
        }
        for _ in 0..25 {
            system.step(0.005);
        }
        let mut digest = TrajectoryDigest::new();
        digest.update_f64_slice(system.u());
        digest.update_f64_slice(system.v());
        (system.u().to_vec(), digest.finish_hex())
    };

    let (u1, h1) = run(42);
    let (u2, h2) = run(42);
    let (u3, h3) = run(43);
    assert_eq!(u1, u2);
    assert_eq!(h1, h2);
    assert_ne!(u1, u3);
    assert_ne!(h1, h3);
    assert_eq!(h1, PINNED_TURING_DIGEST);
}
