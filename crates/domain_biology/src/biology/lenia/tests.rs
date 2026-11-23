use super::presets::stamp_species_at;
use super::*;
use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;
use oxidize_core::rng::OxidizeRng;
use rand::Rng;

#[test]
#[verified_engine::verified]
fn test_kernel_shells_bounds_and_unit_l1_normalization() {
    let clamped_low = KernelShells::new(1.0, &[]);
    assert_eq!(clamped_low.radius(), MIN_KERNEL_RADIUS);
    assert_eq!(clamped_low.ring_count(), 1);
    assert_eq!(clamped_low.ring_weights(), &[1.0]);

    let clamped_high = KernelShells::new(99.0, &[0.5, 1.0, 0.75, 0.25, 0.9]);
    assert_eq!(clamped_high.radius(), MAX_KERNEL_RADIUS);
    assert_eq!(clamped_high.ring_count(), MAX_KERNEL_RINGS);

    let mut spatial = vec![Complex::new(0.0, 0.0); 64 * 64];
    let raw_sum = clamped_high.write_wrapped_spatial_kernel(64, 64, &mut spatial);
    assert!(raw_sum > 10.0);

    let norm_sum: f64 = spatial.iter().map(|c| c.re).sum();
    assert!((norm_sum - 1.0).abs() < 1e-12);
    assert!(spatial.iter().all(|c| c.re >= 0.0 && c.im == 0.0));
}

#[test]
#[verified_engine::verified]
fn test_growth_gaussian_properties() {
    let growth = GrowthParams::new(0.15, 0.015);
    assert!((growth.evaluate(0.15) - 1.0).abs() < 1e-12);
    assert!((growth.evaluate(0.95) + 1.0).abs() < 1e-9);

    let curve = growth.sample_curve(65);
    assert_eq!(curve.len(), 65);
    assert!(
        curve
            .iter()
            .all(|p| p[1].is_finite() && (-1.0..=1.0).contains(&p[1]))
    );
}

#[verified_engine::verified]
fn direct_circular_convolution(
    state: &[f64],
    kernel: &[Complex<f64>],
    w: usize,
    h: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for ky in 0..h {
                let sy = (y + h - ky) % h;
                for kx in 0..w {
                    let sx = (x + w - kx) % w;
                    let a = state[flatten_2d_index(sx, sy, w)];
                    let k = kernel[flatten_2d_index(kx, ky, w)].re;
                    acc += a * k;
                }
            }
            out[flatten_2d_index(x, y, w)] = acc;
        }
    }
    out
}

#[test]
#[verified_engine::verified]
fn test_fft_convolution_matches_direct_and_is_shift_equivariant() {
    let (w, h) = (32, 32);
    let shells = KernelShells::new(6.0, &[1.0, 0.5]);
    let growth = GrowthParams::new(0.18, 0.03);
    let mut engine_a = LeniaEngine::new(w, h, shells, growth, 0.1);
    let mut engine_b = LeniaEngine::new(w, h, shells, growth, 0.1);

    let mut rng = OxidizeRng::new(12345);
    let (shift_x, shift_y) = (9usize, 21usize);
    for y in 0..h {
        for x in 0..w {
            let val = rng.gen_range(0.0..1.0);
            engine_a.state_mut()[flatten_2d_index(x, y, w)] = val;
            let tx = (x + shift_x) % w;
            let ty = (y + shift_y) % h;
            engine_b.state_mut()[flatten_2d_index(tx, ty, w)] = val;
        }
    }

    let mut spatial_k = vec![Complex::new(0.0, 0.0); w * h];
    shells.write_wrapped_spatial_kernel(w, h, &mut spatial_k);
    let expected_pot = direct_circular_convolution(engine_a.state(), &spatial_k, w, h);

    engine_a.step();
    engine_b.step();

    for y in 0..h {
        for x in 0..w {
            let idx_a = flatten_2d_index(x, y, w);
            let idx_b = flatten_2d_index((x + shift_x) % w, (y + shift_y) % h, w);
            assert!((engine_a.potential()[idx_a] - expected_pot[idx_a]).abs() < 1e-11);
            assert!((engine_a.state()[idx_a] - engine_b.state()[idx_b]).abs() < 1e-11);
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_orbium_glides_across_torus_and_heals_from_brush_cut() {
    let mut rng = OxidizeRng::new(7);
    let mut engine = LeniaEngine::from_species(64, 64, LeniaSpecies::Orbium, 0.0, &mut rng);
    let initial_cx = engine.telemetry().centroid_x;
    let initial_cy = engine.telemetry().centroid_y;

    for _ in 0..80 {
        engine.step();
    }
    let mid = engine.telemetry();
    let disp = (mid.centroid_x - initial_cx).hypot(mid.centroid_y - initial_cy);
    assert!(
        disp > 3.5,
        "Orbium should translate across the torus, moved {disp}"
    );
    assert!(mid.centroid_speed > 0.05);
    assert!(mid.total_mass > 25.0 && mid.total_mass < 500.0);

    // Slice part of the organism with a negative brush stroke and verify self-healing survival.
    let cut_x = ((mid.centroid_x as usize) + 3) % 64;
    let cut_y = (mid.centroid_y as usize) % 64;
    engine.paint_disc(cut_x, cut_y, 3.0, -0.6);
    for _ in 0..60 {
        engine.step();
    }
    let healed = engine.telemetry();
    assert!(
        healed.total_mass > 20.0,
        "Orbium should survive tail slice, mass={}",
        healed.total_mass
    );
    assert!(engine.state().iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
#[verified_engine::verified]
fn test_all_species_genomes_and_deterministic_rng_seeding() {
    for species in LeniaSpecies::ALL {
        let mut rng_a = OxidizeRng::new(99);
        let mut rng_b = OxidizeRng::new(99);
        let mut rng_c = OxidizeRng::new(100);

        let mut eng_a = LeniaEngine::from_species(64, 64, species, 0.03, &mut rng_a);
        let mut eng_b = LeniaEngine::from_species(64, 64, species, 0.03, &mut rng_b);
        let eng_c = LeniaEngine::from_species(64, 64, species, 0.03, &mut rng_c);

        assert_eq!(eng_a.state(), eng_b.state());
        assert_ne!(eng_a.state(), eng_c.state());

        for _ in 0..80 {
            eng_a.step();
            eng_b.step();
        }
        assert_eq!(eng_a.state(), eng_b.state());
        let tel = eng_a.telemetry();
        assert!(
            tel.total_mass > 15.0,
            "{:?} died out: mass={}",
            species,
            tel.total_mass
        );
        assert!(
            tel.total_mass < 3500.0,
            "{:?} saturated grid: mass={}",
            species,
            tel.total_mass
        );
    }

    let mut rng = OxidizeRng::new(5);
    let mut engine = LeniaEngine::from_species(64, 64, LeniaSpecies::Orbium, 0.0, &mut rng);
    let before = engine.telemetry().total_mass;
    stamp_species_at(&mut engine, LeniaSpecies::Scutium, 12, 12, 0.01, &mut rng);
    assert!(engine.telemetry().total_mass > before);
}

#[test]
#[verified_engine::verified]
fn test_compiled_rules_clamp_non_finite_and_sustain_polynomial_orbium() {
    use math_commons::ast::{CompiledExpr, parse};

    let chan_poly = parse("2 * max(0, 1 - ((u - mu) / (3 * sigma))^2)^4 - 1").expect("parse");
    let compiled_g = CompiledExpr::compile_with_vars(&chan_poly, &["u"]).expect("compile g");
    assert_eq!(compiled_g.param_names(), &["mu", "sigma"]);
    let growth_rule = CompiledGrowthRule::new(compiled_g, &[0.15, 0.015]);
    assert!((growth_rule.evaluate(0.15) - 1.0).abs() < 1e-12);
    assert!((growth_rule.evaluate(0.0) + 1.0).abs() < 1e-12);

    let singular_g =
        CompiledExpr::compile_with_vars(&parse("1 / (u - 0.25)").expect("parse"), &["u"])
            .expect("compile");
    let singular_rule = CompiledGrowthRule::new(singular_g, &[]);
    for u in [0.0, 0.25, 0.5, f64::NAN, f64::INFINITY] {
        let val = singular_rule.evaluate(u);
        assert!(val.is_finite() && (-1.0..=1.0).contains(&val));
    }

    let poly_shell = CompiledExpr::compile_with_vars(
        &parse("max(0, 1 - ((r - 0.5) / 0.22)^2)^4").expect("parse"),
        &["r"],
    )
    .expect("compile k");
    let kernel_rule = CompiledKernelRule::new(poly_shell, &[]);
    let mut spatial = vec![Complex::new(0.0, 0.0); 64 * 64];
    let raw_sum = kernel_rule.write_wrapped_spatial_kernel(64, 64, 13.0, &mut spatial);
    assert!(raw_sum > 5.0);
    let norm_sum: f64 = spatial.iter().map(|c| c.re).sum();
    assert!((norm_sum - 1.0).abs() < 1e-12);

    let mut rng = OxidizeRng::new(42);
    let mut engine = LeniaEngine::from_species(128, 128, LeniaSpecies::Orbium, 0.0, &mut rng);
    assert!(
        (engine.max_potential() - 0.225).abs() < 0.035,
        "initial Orbium max(U)={} expected near 0.225",
        engine.max_potential()
    );
    engine.set_custom_growth(Some(growth_rule));
    for _ in 0..80 {
        engine.step();
    }
    assert!(
        engine.telemetry().total_mass > 20.0 && engine.telemetry().centroid_speed > 0.05,
        "Chan polynomial Orbium should sustain translation: {:?}",
        engine.telemetry()
    );
}

#[test]
#[verified_engine::verified]
fn test_fft_probe_32_convolution_accuracy_and_regime_classification() {
    let shells = KernelShells::new(6.5, &[1.0, 0.5]);
    let mut state = [0.0_f64; PROBE_CELLS];
    seed_probe_species_32(LeniaSpecies::Orbium, 13.0, &mut state);

    let mut spatial_k = [Complex::new(0.0, 0.0); PROBE_CELLS];
    shells.write_wrapped_spatial_kernel(PROBE_SIDE, PROBE_SIDE, &mut spatial_k);
    let expected = direct_circular_convolution(&state, &spatial_k, PROBE_SIDE, PROBE_SIDE);

    let mut k_hat = spatial_k;
    fft_2d_forward_32(&mut k_hat);
    for c in &mut k_hat {
        *c *= 1.0 / (PROBE_CELLS as f64);
    }
    let mut freq = [Complex::new(0.0, 0.0); PROBE_CELLS];
    for (dst, &a) in freq.iter_mut().zip(state.iter()) {
        *dst = Complex::new(a, 0.0);
    }
    fft_2d_forward_32(&mut freq);
    for (f, &kh) in freq.iter_mut().zip(k_hat.iter()) {
        *f *= kh;
    }
    fft_2d_inverse_32(&mut freq);
    for idx in 0..PROBE_CELLS {
        assert!((freq[idx].re - expected[idx]).abs() < 1e-12);
    }

    let make_spec = |mu: f64, sigma: f64| LeniaProbeSpec {
        species: LeniaSpecies::Orbium,
        seed_radius: 13.0,
        kernel_radius: 13.0,
        ring_weights: &[1.0],
        growth: GrowthParams::new(mu, sigma),
        dt: 0.1,
        custom_growth: None,
        custom_kernel: None,
    };
    assert_eq!(
        evaluate_lenia_probe_32(&make_spec(0.15, 0.015)).outcome,
        LeniaProbeOutcome::LocalizedSoliton
    );
    assert_eq!(
        evaluate_lenia_probe_32(&make_spec(0.28, 0.015)).outcome,
        LeniaProbeOutcome::Extinction
    );
    assert_eq!(
        evaluate_lenia_probe_32(&make_spec(0.10, 0.055)).outcome,
        LeniaProbeOutcome::OvergrownBlob
    );
    assert_eq!(
        evaluate_lenia_probe_32(&make_spec(0.02, 0.04)).outcome,
        LeniaProbeOutcome::VacuumExplosion
    );
}
