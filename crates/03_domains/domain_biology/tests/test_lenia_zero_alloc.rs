//! Verifies that `LeniaEngine::step` performs zero dynamic heap allocations on a `256x256`
//! periodic grid under `verified_engine::allocator::VerifiedAllocator`.

use domain_biology::biology::lenia::{
    CompiledGrowthRule, CompiledKernelRule, GrowthParams, LeniaEngine, LeniaProbeOutcome,
    LeniaProbeSpec, LeniaSpecies, evaluate_lenia_probe_32,
};
use domain_biology::biology::morphogenesis::GrayScottKinetics;
use domain_biology::biology::morphogenesis::ReactionKinetics;
use math_commons::ast::{CompiledExpr, parse};
use oxidize_core::rng::OxidizeRng;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_lenia_256x256_step_zero_heap_allocations() {
    let mut rng = OxidizeRng::new(2026);
    let mut engine = LeniaEngine::from_species(256, 256, LeniaSpecies::Orbium, 0.02, &mut rng);

    let (telemetry, metrics) = VerifiedEngine::run_verified(|| {
        engine.step();
        engine.step();
        engine.telemetry()
    });

    assert_eq!(telemetry.step_count, 2);
    assert!(telemetry.total_mass > 20.0);
    assert!(metrics.function_calls > 0);
}

#[test]
#[verified_engine::verified]
fn test_gray_scott_kinetics_presets_and_zero_alloc_reaction() {
    let mitosis = GrayScottKinetics::mitosis();
    let coral = GrayScottKinetics::coral();
    assert_ne!(mitosis, coral);

    let (rates, _) = VerifiedEngine::run_verified(|| mitosis.reaction([0.5, 0.25]));
    assert!(rates[0].is_finite() && rates[1].is_finite());
    assert!(rates[0] < 0.0);
    assert!(rates[1] > 0.0);
}

#[test]
#[verified_engine::verified]
fn test_lenia_custom_rules_and_probe_32_zero_heap_allocations() {
    let g_expr = parse("2 * max(0, 1 - ((u - mu) / (3 * sigma))^2)^4 - 1").expect("parse g");
    let g_rule = CompiledGrowthRule::new(
        CompiledExpr::compile_with_vars(&g_expr, &["u"]).expect("compile g"),
        &[0.15, 0.015],
    );
    let k_expr = parse("max(0, 1 - ((r - 0.5) / w)^2)^4").expect("parse k");
    let k_rule = CompiledKernelRule::new(
        CompiledExpr::compile_with_vars(&k_expr, &["r"]).expect("compile k"),
        &[0.24],
    );

    let mut rng = OxidizeRng::new(77);
    let mut engine = LeniaEngine::from_species(128, 128, LeniaSpecies::Orbium, 0.0, &mut rng);
    engine.set_custom_growth(Some(g_rule.clone()));
    engine.set_custom_kernel(Some(k_rule.clone()));

    let g_params = [0.15_f64, 0.015_f64];
    let k_params = [0.24_f64];
    let ring_weights = [1.0_f64];

    let ((tel, probe_gauss, probe_custom), _) = VerifiedEngine::run_verified(|| {
        engine.step();
        let gauss_spec = LeniaProbeSpec {
            species: LeniaSpecies::Orbium,
            seed_radius: 13.0,
            kernel_radius: 13.0,
            ring_weights: &ring_weights,
            growth: GrowthParams::new(0.15, 0.015),
            dt: 0.1,
            custom_growth: None,
            custom_kernel: None,
        };
        let custom_spec = LeniaProbeSpec {
            species: LeniaSpecies::Orbium,
            seed_radius: 13.0,
            kernel_radius: 13.0,
            ring_weights: &ring_weights,
            growth: GrowthParams::new(0.15, 0.015),
            dt: 0.1,
            custom_growth: Some((&g_rule, &g_params)),
            custom_kernel: Some((&k_rule, &k_params)),
        };
        (
            engine.telemetry(),
            evaluate_lenia_probe_32(&gauss_spec),
            evaluate_lenia_probe_32(&custom_spec),
        )
    });

    assert!(tel.total_mass > 15.0);
    assert_eq!(probe_gauss.outcome, LeniaProbeOutcome::LocalizedSoliton);
    assert_eq!(probe_custom.outcome, LeniaProbeOutcome::LocalizedSoliton);
}
