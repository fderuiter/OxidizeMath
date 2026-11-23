//! Cross-platform deterministic tests for pure-Rust `libm` (`soft_float`) wrappers in `math_commons`.

use math_commons::math_kernel::soft_float;
use oxidize_core::rng::TrajectoryDigest;

const PINNED_SOFT_FLOAT_F64_DIGEST: &str =
    "b2a2af0384d59d48b411f396e570065c88719030109f49f019d8a52cccf6feff";
const PINNED_SOFT_FLOAT_F32_DIGEST: &str =
    "d3de5d0d7273cbb6260520c605ebdac078b4917778ce88e8e2cd5c5cba6fd754";

#[test]
#[verified_engine::verified]
fn test_soft_float_f64_grid_pinned_digest() {
    let inputs = [0.05_f64, 0.25, 0.75, 1.0, 1.75, 3.125, 7.5];
    let mut digest = TrajectoryDigest::new();
    for &x in &inputs {
        let (s, c) = soft_float::sincos(x);
        assert_eq!(s.to_bits(), soft_float::sin(x).to_bits());
        assert_eq!(c.to_bits(), soft_float::cos(x).to_bits());
        let row = [
            soft_float::ln(x),
            soft_float::ln_1p(x),
            soft_float::log2(x),
            soft_float::log10(x),
            soft_float::exp(-x),
            soft_float::exp_m1(x * 0.1),
            soft_float::powf(x, 0.65),
            soft_float::sqrt(x),
            soft_float::cbrt(x - 1.0),
            soft_float::hypot(x, 1.25),
            s,
            c,
            soft_float::tan(x * 0.2),
            soft_float::atan2(x - 1.0, 0.75),
            soft_float::tanh(x - 1.5),
        ];
        digest.update_f64_slice(&row);
    }
    assert_eq!(digest.finish_hex(), PINNED_SOFT_FLOAT_F64_DIGEST);
}

#[test]
#[verified_engine::verified]
fn test_soft_float_f32_grid_pinned_digest() {
    let inputs = [0.1_f32, 0.35, 0.9, 1.5, 2.75, 5.25];
    let mut digest = TrajectoryDigest::new();
    for &x in &inputs {
        let (s, c) = soft_float::sincosf(x);
        assert_eq!(s.to_bits(), soft_float::sinf(x).to_bits());
        assert_eq!(c.to_bits(), soft_float::cosf(x).to_bits());
        let row = [
            soft_float::lnf(x),
            soft_float::expf(-x),
            soft_float::powff(x, 0.75),
            soft_float::sqrtf(x),
            soft_float::hypotf(x, 1.5),
            s,
            c,
            soft_float::tanf(x * 0.2),
            soft_float::atan2f(x - 1.0, 0.5),
        ];
        digest.update_f32_slice(&row);
    }
    assert_eq!(digest.finish_hex(), PINNED_SOFT_FLOAT_F32_DIGEST);
}
