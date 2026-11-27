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

struct ConstWordRng(u64);

impl RngCore for ConstWordRng {
    fn next_u32(&mut self) -> u32 {
        self.0 as u32
    }

    fn next_u64(&mut self) -> u64 {
        self.0
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        dest.fill(0);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

struct Counter24Rng(u32);

impl RngCore for Counter24Rng {
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    fn next_u64(&mut self) -> u64 {
        let cur = self.0;
        self.0 = self.0.wrapping_add(1);
        (cur as u64) << 40
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        dest.fill(0);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

fn marsaglia_normal_cdf(z: f64) -> f64 {
    if z <= -8.5 {
        return 0.0;
    }
    if z >= 8.5 {
        return 1.0;
    }
    let z2 = z * z;
    let mut term = z;
    let mut sum = z;
    for k in 1..=100_u32 {
        term *= z2 / f64::from(2 * k + 1);
        sum += term;
    }
    0.5 + sum * (-0.5 * z2).exp() / (2.0 * core::f64::consts::PI).sqrt()
}

#[test]
fn test_ln_open_unit_one_ulp_and_f64_endpoint_bounds() {
    use crate::RngScalar;
    use crate::scalar::ln_open_unit_f64;

    let min_u: f64 = sample_open(&mut ConstWordRng(0));
    let max_u: f64 = sample_open(&mut ConstWordRng(u64::MAX));
    assert_eq!(min_u, 2.0_f64.powi(-54));
    assert_eq!(max_u, 1.0 - 2.0_f64.powi(-53));
    assert_eq!(<f64 as RngScalar>::from_dyadic_53_open(0), min_u);
    assert_eq!(
        <f64 as RngScalar>::from_dyadic_53_open((1_u64 << 53) - 1),
        max_u
    );

    let anchors = [
        f64::from_bits(1),
        min_u,
        2.0_f64.powi(-53),
        1.0e-15,
        0.5 - f64::EPSILON,
        0.5,
        0.5 + f64::EPSILON,
        core::f64::consts::FRAC_1_SQRT_2,
        max_u,
    ];
    for u in anchors {
        let ulps = ln_open_unit_f64(u).to_bits().abs_diff(u.ln().to_bits());
        assert!(ulps <= 1, "ln_open_unit_f64({u:e}) differed by {ulps} ULPs");
    }

    let mut rng = OxidizeRng::new(0x1DEA_2026);
    for _ in 0..16_384 {
        let u: f64 = sample_open(&mut rng);
        let ulps = ln_open_unit_f64(u).to_bits().abs_diff(u.ln().to_bits());
        assert!(ulps <= 1, "ln_open_unit_f64({u:e}) differed by {ulps} ULPs");
    }
}

#[test]
fn test_probit_tail_signs_monotonicity_and_ks_normality() {
    use crate::scalar::acklam_probit_f64;

    let z_lo = acklam_probit_f64(2.0_f64.powi(-54));
    let z_hi = acklam_probit_f64(1.0 - 2.0_f64.powi(-53));
    assert!((z_lo - (-8.292_361_075_813_595)).abs() < 1.0e-12);
    assert!((z_hi - 8.209_536_151_601_386).abs() < 1.0e-12);
    assert!(acklam_probit_f64(0.01) < -2.326 && acklam_probit_f64(0.02424) < -1.973);
    assert!(acklam_probit_f64(0.97576) > 1.973 && acklam_probit_f64(0.99) > 2.326);

    for &seam in &[0.02425, 0.075, 0.925, 0.97575, (-25.0_f64).exp()] {
        let delta = seam.min(1.0 - seam) * 1.0e-12;
        let z_lo_s = acklam_probit_f64(seam - delta);
        let z_hi_s = acklam_probit_f64(seam + delta);
        let gap = z_hi_s - z_lo_s;
        assert!(
            gap > 0.0 && gap < 1.0e-9,
            "discontinuity across p={seam}: gap={gap}"
        );
    }

    let mut prev = z_lo;
    for idx in 1..4_096_u32 {
        let p = f64::from(idx) / 4_096.0;
        let z = acklam_probit_f64(p);
        assert!(z > prev, "non-monotonic probit at p={p}: {prev} >= {z}");
        prev = z;
    }

    let mut rng = OxidizeRng::new(0x5150_2026);
    let mut draws = std::vec::Vec::with_capacity(8_192);
    for _ in 0..8_192 {
        draws.push(sample_standard_normal_f64(&mut rng));
    }
    draws.sort_unstable_by(f64::total_cmp);
    let n = draws.len() as f64;
    let mut ks_d = 0.0_f64;
    for (i, &z) in draws.iter().enumerate() {
        let cdf = marsaglia_normal_cdf(z);
        let d_lo = ((i as f64) / n - cdf).abs();
        let d_hi = (((i + 1) as f64) / n - cdf).abs();
        ks_d = ks_d.max(d_lo.max(d_hi));
    }
    assert!(ks_d < 0.02, "KS statistic D={ks_d} rejected N(0, 1)");
}

#[test]
fn test_f32_sample_open_exhaustive_24bit_grid() {
    let mut counter = Counter24Rng(0);
    let mut prev = 0.0_f32;
    let mut upper_half = 0_u32;
    let total = 1_u32 << 24;

    for bits in 0..total {
        let u: f32 = sample_open(&mut counter);
        if bits == 0 {
            assert_eq!(u, 2.0_f32.powi(-25));
        } else if bits == 1 {
            assert_eq!(u, 2.0_f32.powi(-24));
        } else if bits == total - 1 {
            assert_eq!(u.to_bits(), 0x3f7f_ffff);
        }
        assert!(
            u > prev && u < 1.0,
            "collapse at bits={bits}: {prev} -> {u}"
        );
        if u >= 0.5 {
            upper_half += 1;
        }
        prev = u;
    }
    assert_eq!(upper_half, 1_u32 << 23);
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

    let golden = r#"{"seed":[4,8,15,16,23,42,4,8,15,16,23,42,4,8,15,16,23,42,4,8,15,16,23,42,4,8,15,16,23,42,4,8],"stream":27182818284,"word_pos":314159265359}"#;
    let mut from_golden: OxidizeRng =
        serde_json::from_str(golden).expect("deserialize golden rand_chacha payload");
    assert_eq!(from_golden.stream(), 27_182_818_284);
    assert_eq!(from_golden.word_pos(), 314_159_265_359);
    assert_eq!(
        serde_json::to_string(&from_golden).expect("re-serialize golden OxidizeRng"),
        golden
    );

    let seq_payload = r#"[[4,8,15,16,23,42,4,8,15,16,23,42,4,8,15,16,23,42,4,8,15,16,23,42,4,8,15,16,23,42,4,8],27182818284,314159265359]"#;
    let mut from_seq: OxidizeRng =
        serde_json::from_str(seq_payload).expect("deserialize positional ChaCha8Rng tuple");
    assert_eq!(from_golden, from_seq);
    assert_eq!(from_golden.next_u64(), from_seq.next_u64());
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
