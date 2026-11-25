//! Unit tests for 64D `SobolView`, `HaltonView`, `NiederreiterView`, static table budgets,
//! `VerifiedAllocator` zero-allocation locks, and Halton collinearity removal.
#![cfg_attr(any(), verified(opt_out = "tests"))]

use crate::qmc::{
    FIRST_64_PRIMES, HaltonView, MAX_HALTON_DIM, MAX_NIEDERREITER_DIM, MAX_SOBOL_DIM,
    NiederreiterView, QmcError, SobolView, radical_inverse, total_qmc_static_table_bytes,
};
use crate::{BASE_SEED, OxidizeRng, owen_scramble_53};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static GLOBAL_VERIFIED_ALLOC: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_static_qmc_rodata_stays_strictly_below_16_kib() {
    let bytes = total_qmc_static_table_bytes();
    assert!(
        bytes < 16 * 1024,
        "static QMC tables exceeded 16 KiB budget: {bytes} bytes"
    );
    assert!(bytes > 4 * 1024);
}

#[test]
fn test_owen_scramble_53_bijection_on_dyadic_strata() {
    let mut seen = [false; 64];
    for top6 in 0..64_u64 {
        let raw_53 = top6 << (53 - 6);
        let scrambled = owen_scramble_53(raw_53, BASE_SEED, 37);
        let bucket = (scrambled >> (53 - 6)) as usize;
        assert!(
            !seen[bucket],
            "Owen scramble collided on 2^-6 dyadic stratum {bucket}"
        );
        seen[bucket] = true;
    }
    assert!(seen.iter().all(|&b| b));
}

#[test]
fn test_sobol_and_niederreiter_64d_incremental_match_point_at() {
    let mut s_state = [0_u64; MAX_SOBOL_DIM];
    let mut s_shifts = [0.0_f64; MAX_SOBOL_DIM];
    let mut sobol = SobolView::new(MAX_SOBOL_DIM, &mut s_state, &mut s_shifts)
        .expect("64D SobolView")
        .with_scramble_seed(Some(42));

    let mut n_state = [0_u64; MAX_NIEDERREITER_DIM];
    let mut n_shifts = [0.0_f64; MAX_NIEDERREITER_DIM];
    let mut nied = NiederreiterView::new(MAX_NIEDERREITER_DIM, &mut n_state, &mut n_shifts)
        .expect("64D NiederreiterView")
        .with_scramble_seed(Some(42));

    let mut stepped = [0.0_f64; 64];
    let mut direct = [0.0_f64; 64];
    for idx in 0..128_u64 {
        sobol.next_point(&mut stepped).expect("sobol step");
        sobol.point_at(idx, &mut direct).expect("sobol direct");
        assert_eq!(stepped, direct, "Sobol mismatch at idx={idx}");
        assert!(stepped.iter().all(|&v| v > 0.0 && v < 1.0));

        nied.next_point(&mut stepped).expect("niederreiter step");
        nied.point_at(idx, &mut direct)
            .expect("niederreiter direct");
        assert_eq!(stepped, direct, "Niederreiter mismatch at idx={idx}");
        assert!(stepped.iter().all(|&v| v > 0.0 && v < 1.0));
    }
}

#[test]
fn test_views_64d_f32_and_f64_zero_allocation_under_verified_allocator() {
    let mut s_state = [0_u64; 64];
    let mut s_shifts = [0.0_f64; 64];
    let mut n_state = [0_u64; 64];
    let mut n_shifts = [0.0_f64; 64];
    let mut h_shifts = [0.0_f64; 64];
    let mut out_f64 = [0.0_f64; 64];
    let mut out_f32 = [0.0_f32; 64];
    let mut rng = OxidizeRng::new(BASE_SEED);

    let (acc, _) = VerifiedEngine::run_verified(|| {
        let mut sobol = SobolView::new(64, &mut s_state, &mut s_shifts)
            .expect("64D sobol")
            .with_scramble_seed(Some(7));
        let mut nied = NiederreiterView::new(64, &mut n_state, &mut n_shifts)
            .expect("64D nied")
            .with_scramble_seed(Some(11));
        let mut halton = HaltonView::new(64, &mut h_shifts).expect("64D halton");

        sobol.apply_cranley_patterson_shift(&mut rng);
        nied.apply_cranley_patterson_shift(&mut rng);
        halton.apply_cranley_patterson_shift(&mut rng);

        let mut total = 0.0_f64;
        for _ in 0..64 {
            sobol.next_point(&mut out_f64).expect("sobol f64");
            sobol.next_point(&mut out_f32).expect("sobol f32");
            nied.next_point(&mut out_f64).expect("nied f64");
            nied.next_point(&mut out_f32).expect("nied f32");
            halton.next_point(&mut out_f64).expect("halton f64");
            halton.next_point(&mut out_f32).expect("halton f32");
            assert!(out_f32.iter().all(|&x| x > 0.0 && x < 1.0));
            total += out_f64[63] + (out_f32[63] as f64);
        }
        total
    });
    assert!(acc > 0.0 && acc.is_finite());
}

fn pearson_r_and_occ16(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut dx = 0.0;
    let mut dy = 0.0;
    let mut grid = [false; 256];
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        num += (x - mx) * (y - my);
        dx += (x - mx) * (x - mx);
        dy += (y - my) * (y - my);
        let gx = ((x * 16.0) as usize).min(15);
        let gy = ((y * 16.0) as usize).min(15);
        grid[gy * 16 + gx] = true;
    }
    let occ = (grid.iter().filter(|&&c| c).count() as f64) / 256.0;
    (num / (dx.sqrt() * dy.sqrt()), occ)
}

#[test]
fn test_scrambled_halton_eliminates_collinearity_on_axes_62_to_64() {
    assert_eq!(FIRST_64_PRIMES[61], 293);
    assert_eq!(FIRST_64_PRIMES[62], 307);
    assert_eq!(FIRST_64_PRIMES[63], 311);

    let mut shifts = [0.0_f64; MAX_HALTON_DIM];
    let mut raw_view = HaltonView::new(MAX_HALTON_DIM, &mut shifts)
        .expect("64D")
        .raw_unpermuted();
    let mut r61 = [0.0_f64; 256];
    let mut r62 = [0.0_f64; 256];
    let mut pt = [0.0_f64; MAX_HALTON_DIM];
    for i in 0..256 {
        raw_view.next_point(&mut pt).expect("raw point");
        r61[i] = pt[61];
        r62[i] = pt[62];
    }
    let (raw_corr, raw_occ) = pearson_r_and_occ16(&r61, &r62);
    assert!(raw_corr > 0.999, "unscrambled Halton r={raw_corr}");
    assert!(raw_occ < 0.12, "unscrambled occupancy={raw_occ}");

    let mut scrambled = HaltonView::new(MAX_HALTON_DIM, &mut shifts).expect("64D scrambled");
    let mut a61 = [0.0_f64; 256];
    let mut a62 = [0.0_f64; 256];
    let mut a63 = [0.0_f64; 256];
    for i in 0..256 {
        scrambled.next_point(&mut pt).expect("scrambled point");
        a61[i] = pt[61];
        a62[i] = pt[62];
        a63[i] = pt[63];
    }
    for (pair, (xs, ys)) in [
        ("62-63", (&a61[..], &a62[..])),
        ("63-64", (&a62[..], &a63[..])),
        ("62-64", (&a61[..], &a63[..])),
    ] {
        let (r, occ) = pearson_r_and_occ16(xs, ys);
        assert!(r.abs() < 0.15, "axes {pair} |r|={} >= 0.15", r.abs());
        assert!(occ > 0.60, "axes {pair} 16x16 occupancy={occ} <= 0.60");
    }
}

#[test]
fn test_qmc_validation_errors_and_radical_inverse() {
    let mut st = [0_u64; 4];
    let mut sh = [0.0_f64; 4];
    assert_eq!(
        SobolView::new(0, &mut [], &mut []).unwrap_err(),
        QmcError::InvalidDimension {
            dimension: 0,
            max_supported: 64
        }
    );
    assert_eq!(
        NiederreiterView::new(65, &mut st, &mut sh).unwrap_err(),
        QmcError::InvalidDimension {
            dimension: 65,
            max_supported: 64
        }
    );
    assert_eq!(
        radical_inverse(5, 1).unwrap_err(),
        QmcError::InvalidBase { base: 1 }
    );
    assert!((radical_inverse(3, 2).expect("base 2") - 0.75).abs() < 1.0e-15);
}

#[cfg(feature = "alloc")]
#[test]
fn test_owned_alloc_sequences_match_views() {
    let mut sobol = crate::SobolSequence::new(64).expect("64D");
    let mut halton = crate::HaltonSequence::new(64).expect("64D");
    let mut nied = crate::NiederreiterSequence::new(64).expect("64D");
    let mut buf = [0.0_f64; 64];
    sobol.next_point(&mut buf).expect("sobol");
    halton.next_point(&mut buf).expect("halton");
    nied.next_point(&mut buf).expect("nied");
    assert!(buf.iter().all(|&x| x > 0.0 && x < 1.0));
}

#[test]
fn test_native_and_wasm32_bit_exact_stream_and_qmc_digest() {
    use crate::{RngCore, sample_bounded_usize, sample_open};

    let base = OxidizeRng::new(BASE_SEED);
    let mut s1 = base.split_stream(1);
    let mut s2 = base.split_stream(2);
    let mut acc = s1.next_u64() ^ s2.next_u64().rotate_left(13);
    acc ^= sample_open::<f64, _>(&mut s1).to_bits();
    acc ^= (sample_bounded_usize(&mut s2, 1000) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);

    let mut st = [0_u64; 64];
    let mut sh = [0.0_f64; 64];
    let mut pt = [0.0_f64; 64];
    let mut sobol = SobolView::new(64, &mut st, &mut sh)
        .expect("64D sobol")
        .with_scramble_seed(Some(7));
    sobol.apply_cranley_patterson_shift(&mut s1);
    sobol.next_point(&mut pt).expect("sobol step");
    acc ^= pt[63].to_bits();

    let mut nied = NiederreiterView::new(64, &mut st, &mut sh)
        .expect("64D nied")
        .with_scramble_seed(Some(11));
    nied.apply_cranley_patterson_shift(&mut s2);
    nied.next_point(&mut pt).expect("nied step");
    acc ^= pt[63].to_bits().rotate_left(7);

    let mut halton = HaltonView::new(64, &mut sh).expect("64D halton");
    halton.apply_cranley_patterson_shift(&mut s1);
    halton.next_point(&mut pt).expect("halton step");
    let digest = acc ^ pt[63].to_bits().rotate_left(19);
    assert_eq!(digest, 0x59d1_6243_5f14_5673);
}
