//! Cross-platform deterministic tests for `OxidizeRng` streams and IEEE-754 canonical `TrajectoryDigest`.

use oxidize_core::rng::{
    BASE_SEED, CANONICAL_F32_NAN_BITS, CANONICAL_F64_NAN_BITS, OxidizeRng, TrajectoryDigest,
    canonical_f32_bits, canonical_f64_bits,
};
use rand::RngCore;

const PINNED_RNG_STREAM_DIGEST: &str =
    "d5681d20917cfcac0ab483fe0495d8a462c0ad6677f48e719c3536aeaf7c0097";
const PINNED_CANONICAL_FLOAT_DIGEST: &str =
    "00719090c543608359bb73b4fd9ff43685fcc41ed4298412ecad70512f414774";

#[test]
fn test_oxidize_rng_chacha8_stream_pinned_digest() {
    let sample_stream = |seed: u64| -> (Vec<u64>, String) {
        let mut rng = OxidizeRng::from_seed(seed);
        let mut digest = TrajectoryDigest::new();
        let mut words = Vec::with_capacity(32);
        for idx in 0..32_usize {
            let word = rng.next_u64();
            let low = rng.next_u32();
            let mut block = [0_u8; 13];
            rng.fill_bytes(&mut block);
            digest.update_usize(idx);
            digest.update_u64(word);
            digest.update_u32(low);
            digest.update_bytes(&block);
            words.push(word);
        }
        (words, digest.finish_hex())
    };

    let (w_a, hex_a) = sample_stream(BASE_SEED ^ 0x0007);
    let (w_b, hex_b) = sample_stream(BASE_SEED ^ 0x0007);
    let (w_c, hex_c) = sample_stream(BASE_SEED ^ 0x0008);

    assert_eq!(w_a, w_b);
    assert_eq!(hex_a, hex_b);
    assert_ne!(w_a, w_c);
    assert_ne!(hex_a, hex_c);
    assert_eq!(hex_a, PINNED_RNG_STREAM_DIGEST);
}

#[test]
fn test_trajectory_digest_canonicalizes_signed_zero_and_nan_payloads() {
    assert_eq!(canonical_f64_bits(-0.0_f64), 0);
    assert_eq!(canonical_f64_bits(0.0_f64), 0);
    assert_eq!(canonical_f32_bits(-0.0_f32), 0);
    assert_eq!(canonical_f32_bits(0.0_f32), 0);

    let nan64_neg = f64::from_bits(0xfff8_0000_0000_0001);
    let nan64_payload = f64::from_bits(0x7ff8_dead_beef_cafe);
    assert_eq!(canonical_f64_bits(nan64_neg), CANONICAL_F64_NAN_BITS);
    assert_eq!(canonical_f64_bits(nan64_payload), CANONICAL_F64_NAN_BITS);

    let nan32_neg = f32::from_bits(0xffc0_0001);
    let nan32_payload = f32::from_bits(0x7fc5_4321);
    assert_eq!(canonical_f32_bits(nan32_neg), CANONICAL_F32_NAN_BITS);
    assert_eq!(canonical_f32_bits(nan32_payload), CANONICAL_F32_NAN_BITS);

    let mut d_pos = TrajectoryDigest::new();
    d_pos.update_f64_slice(&[0.0, 1.5, -2.25, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    d_pos.update_f32_slice(&[0.0, -3.5, f32::NAN]);
    d_pos.update_i64(-42);

    let mut d_alt = TrajectoryDigest::new();
    d_alt.update_f64_slice(&[
        -0.0,
        1.5,
        -2.25,
        f64::INFINITY,
        f64::NEG_INFINITY,
        nan64_neg,
    ]);
    d_alt.update_f32_slice(&[-0.0, -3.5, nan32_payload]);
    d_alt.update_i64(-42);

    let hex_pos = d_pos.finish_hex();
    let hex_alt = d_alt.finish_hex();
    assert_eq!(hex_pos, hex_alt);
    assert_eq!(hex_pos, PINNED_CANONICAL_FLOAT_DIGEST);
}
