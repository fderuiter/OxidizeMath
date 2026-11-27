//! Unit tests for `OxidizeRng`, ChaCha8 streams, word seeking, and generic `RngScalar` samplers.
#![cfg_attr(any(), verified(opt_out = "tests"))]

use crate::{
    BASE_SEED, OxidizeRng, RngCore, SeedableRng, fill_cranley_patterson_shifts, sample_bounded_u64,
    sample_bounded_usize, sample_closed_open, sample_open, sample_standard_normal,
    sample_standard_normal_f64, sample_symmetric_open,
};
use verified_engine::engine::VerifiedEngine;

#[test]
fn test_oxidize_rng_determinism_and_seedable_rng() {
    let mut a = OxidizeRng::new(BASE_SEED);
    let mut b = OxidizeRng::from_seed(BASE_SEED);
    let mut c = <OxidizeRng as SeedableRng>::seed_from_u64(BASE_SEED);
    for _ in 0..32 {
        let va = a.next_u64();
        assert_eq!(va, b.next_u64());
        assert_eq!(va, c.next_u64());
    }
    assert_eq!(a.next_u32(), b.next_u32());

    let key = [0x5A_u8; 32];
    let mut k1 = OxidizeRng::from_seed_bytes(key);
    let mut k2 = <OxidizeRng as SeedableRng>::from_seed(key);
    let mut buf1 = [0_u8; 24];
    let mut buf2 = [0_u8; 24];
    k1.fill_bytes(&mut buf1);
    k2.try_fill_bytes(&mut buf2).expect("infallible ChaCha8");
    assert_eq!(buf1, buf2);
}

#[test]
fn test_chacha8_orthogonal_streams_and_word_seeking() {
    let base = OxidizeRng::new(42);
    let mut s1 = base.split_stream(1);
    let mut s2 = base.split_stream(2);
    let mut s1_direct = OxidizeRng::new(42).with_stream(1);

    assert_eq!(s1.stream(), 1);
    assert_eq!(s2.stream(), 2);
    let v1 = s1.next_u64();
    let v2 = s2.next_u64();
    assert_ne!(v1, v2);
    assert_eq!(v1, s1_direct.next_u64());

    let mut stepped = OxidizeRng::new(99).with_stream(7);
    let mut jumped = OxidizeRng::new(99).with_stream(7);
    for _ in 0..10 {
        let _ = stepped.next_u64(); // each next_u64 consumes two 32-bit words
    }
    jumped.jump_words(20);
    assert_eq!(stepped.word_pos(), 20);
    assert_eq!(jumped.word_pos(), 20);
    assert_eq!(stepped.next_u64(), jumped.next_u64());

    jumped.set_word_pos(0);
    let mut fresh = OxidizeRng::new(99).with_stream(7);
    assert_eq!(jumped.next_u64(), fresh.next_u64());
}

#[test]
fn test_generic_f32_and_f64_samplers_stay_in_strict_intervals() {
    let mut rng = OxidizeRng::new(2026);
    let (count, _) = VerifiedEngine::run_verified(|| {
        let mut shifts_f32 = [0.0_f32; 16];
        fill_cranley_patterson_shifts(&mut shifts_f32, &mut rng);
        assert!(shifts_f32.iter().all(|&s| (0.0..1.0).contains(&s)));

        let mut n = 0_usize;
        for _ in 0..4_096 {
            let co32: f32 = sample_closed_open(&mut rng);
            let op32: f32 = sample_open(&mut rng);
            let sy32: f32 = sample_symmetric_open(&mut rng);
            assert!((0.0..1.0).contains(&co32));
            assert!(op32 > 0.0 && op32 < 1.0);
            assert!(sy32 > -1.0 && sy32 < 1.0);

            let co64: f64 = sample_closed_open(&mut rng);
            let op64: f64 = rng.uniform_open();
            let sy64: f64 = rng.uniform_symmetric();
            assert!((0.0..1.0).contains(&co64));
            assert!(op64 > 0.0 && op64 < 1.0);
            assert!(sy64 > -1.0 && sy64 < 1.0);
            n += 1;
        }
        n
    });
    assert_eq!(count, 4_096);
}

#[test]
fn test_unbiased_bounded_index_and_standard_normal_moments() {
    use crate::{RngError, try_sample_bounded_u64, try_sample_bounded_usize};

    let mut rng = OxidizeRng::new(777);
    assert_eq!(
        try_sample_bounded_u64(&mut rng, 0),
        Err(RngError::ZeroUpperBound)
    );
    assert_eq!(
        try_sample_bounded_usize(&mut rng, 0),
        Err(RngError::ZeroUpperBound)
    );
    assert_eq!(try_sample_bounded_u64(&mut rng, 1), Ok(0));
    assert_eq!(try_sample_bounded_usize(&mut rng, 1), Ok(0));
    assert_eq!(sample_bounded_u64(&mut rng, 1), 0);
    assert_eq!(sample_bounded_usize(&mut rng, 1), 0);

    let mut counts = [0_u32; 7];
    for _ in 0..7_000 {
        let idx = rng.bounded_usize(7);
        assert!(idx < 7);
        counts[idx] += 1;
    }
    for &c in &counts {
        assert!(
            (850..=1_150).contains(&c),
            "expected ~1000 per bucket, got {c}"
        );
    }

    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let n = 8_192_usize;
    for _ in 0..n {
        let z = sample_standard_normal_f64(&mut rng);
        assert!(z.is_finite());
        sum += z;
        sum_sq += z * z;
    }
    let mean = sum / (n as f64);
    let var = sum_sq / (n as f64) - mean * mean;
    assert!(mean.abs() < 0.05, "normal mean={mean}");
    assert!((var - 1.0).abs() < 0.06, "normal var={var}");
    let z32: f32 = sample_standard_normal(&mut rng);
    assert!(z32.is_finite());
}

#[test]
fn math_rng_serde_json_dev_dep_linked() {
    let v: u64 = serde_json::from_str("42").expect("json u64");
    assert_eq!(v, 42);
}

#[cfg(feature = "serde")]
#[test]
fn test_serde_roundtrip_preserves_exact_stream_and_position() {
    use crate::QmcError;

    let mut rng = OxidizeRng::new(12345).with_stream(9);
    let _ = rng.next_u64();
    let _ = rng.next_u64();
    let json = serde_json::to_string(&rng).expect("serialize OxidizeRng");
    let mut decoded: OxidizeRng = serde_json::from_str(&json).expect("deserialize OxidizeRng");
    assert_eq!(rng, decoded);
    assert_eq!(rng.stream(), decoded.stream());
    assert_eq!(rng.word_pos(), decoded.word_pos());
    assert_eq!(rng.next_u64(), decoded.next_u64());

    let err = QmcError::InvalidDimension {
        dimension: 0,
        max_supported: 64,
    };
    let err_json = serde_json::to_string(&err).expect("serialize QmcError");
    assert_eq!(
        serde_json::from_str::<QmcError>(&err_json).expect("deserialize QmcError"),
        err
    );
}

struct StackDiagRecorder {
    str_fields: usize,
    display_fields: usize,
    last_u32: u32,
    last_usize: usize,
}

impl math_diagnostics::DiagnosticVisitor for StackDiagRecorder {
    fn record_str(&mut self, _key: &str, _value: &str) {
        self.str_fields += 1;
    }

    fn record_display(&mut self, _key: &str, _value: &dyn core::fmt::Display) {
        self.display_fields += 1;
    }

    fn record_u32(&mut self, _key: &str, value: u32) {
        self.last_u32 = value;
        self.display_fields += 1;
    }

    fn record_usize(&mut self, _key: &str, value: usize) {
        self.last_usize = value;
        self.display_fields += 1;
    }
}

#[test]
fn test_rng_and_qmc_errors_visit_diagnostic_fields_with_zero_allocations() {
    use crate::{QmcError, RngError};
    use math_diagnostics::{Diagnostic, Severity};

    let rng_err = RngError::ZeroUpperBound;
    let qmc_variants = [
        QmcError::InvalidDimension {
            dimension: 0,
            max_supported: 64,
        },
        QmcError::InvalidBase { base: 1 },
        QmcError::BufferLengthMismatch {
            expected: 8,
            actual: 4,
        },
        QmcError::StateLengthMismatch {
            expected: 16,
            actual: 9,
        },
    ];

    let ((str_fields, display_fields, last_u32, last_usize), _) =
        VerifiedEngine::run_verified(|| {
            let mut rec = StackDiagRecorder {
                str_fields: 0,
                display_fields: 0,
                last_u32: 0,
                last_usize: 0,
            };
            assert_eq!(rng_err.severity(), Severity::Error);
            rng_err.visit_fields(&mut rec);
            for err in &qmc_variants {
                assert_eq!(err.severity(), Severity::Error);
                err.visit_fields(&mut rec);
            }
            (
                rec.str_fields,
                rec.display_fields,
                rec.last_u32,
                rec.last_usize,
            )
        });

    assert_eq!(str_fields, 10);
    assert_eq!(display_fields, 12);
    assert_eq!(last_u32, 1);
    assert_eq!(last_usize, 9);
    assert_eq!(rng_err.code(), "RNG-001");
    assert_eq!(qmc_variants[0].code(), "QMC-001");
    assert_eq!(qmc_variants[1].code(), "QMC-002");
    assert_eq!(qmc_variants[2].code(), "QMC-003");
    assert_eq!(qmc_variants[3].code(), "QMC-004");
}
