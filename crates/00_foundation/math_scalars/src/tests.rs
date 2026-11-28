// [cite:math_scalars]
//! Unit, 0-ULP IEEE-754 parity, automatic-differentiation, and zero-allocation tests for `math_scalars`.

use crate::{
    Complex32, Complex64, ComplexField, Dimension, Dual, Field, GridIndex, RefOps, Ring, Scalar,
    ScalarError, ScalarWorkspace, SolverWorkspace, StepSize, flatten_2d_index, flatten_3d_index,
    hypot_f32, hypot_f64, sqrt_f32, sqrt_f64,
};

#[global_allocator]
static ALLOCATOR: verified_engine::allocator::VerifiedAllocator =
    verified_engine::allocator::VerifiedAllocator;

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[test]
fn sqrt_f64_achieves_zero_ulp_parity_across_200k_normals_and_subnormals() {
    // Regression check: 5-step Heron previously rounded 3.9999999999999996 up to 2.0 (1-ULP drift).
    let boundary = 3.9999999999999996_f64;
    let expected_boundary = 1.9999999999999998_f64;
    assert_eq!(sqrt_f64(boundary).to_bits(), expected_boundary.to_bits());
    assert_eq!(
        <f64 as ComplexField>::sqrt(boundary).to_bits(),
        expected_boundary.to_bits()
    );

    // Special IEEE-754 values
    assert_eq!(sqrt_f64(0.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(sqrt_f64(-0.0).to_bits(), (-0.0_f64).to_bits());
    assert_eq!(sqrt_f64(f64::INFINITY), f64::INFINITY);
    assert!(sqrt_f64(-1.0).is_nan());
    assert!(sqrt_f64(f64::NAN).is_nan());

    // 200,000 deterministic random positive normal f64 bit patterns across full exponent range 1..=2046
    let mut seed = 0xD1B5_4A32_D192_ED03_u64;
    for _ in 0..200_000 {
        let raw = splitmix64_next(&mut seed);
        let exp_bits = ((raw >> 52) % 2046) + 1;
        let mantissa = raw & ((1_u64 << 52) - 1);
        let x = f64::from_bits((exp_bits << 52) | mantissa);
        assert!(x.is_normal() && x > 0.0);
        let got = <f64 as ComplexField>::sqrt(x);
        let ref_val = libm::sqrt(x);
        assert_eq!(
            got.to_bits(),
            ref_val.to_bits(),
            "non-zero ULP error on normal x={x:e} (bits={:#018x})",
            x.to_bits()
        );
    }

    // Subnormal f64 values (exponent = 0, nonzero mantissa)
    for k in 0..52 {
        let sub = f64::from_bits(1_u64 << k);
        assert!(!sub.is_normal() && sub > 0.0);
        assert_eq!(
            <f64 as ComplexField>::sqrt(sub).to_bits(),
            libm::sqrt(sub).to_bits()
        );
    }
    for _ in 0..4_096 {
        let mantissa = (splitmix64_next(&mut seed) & ((1_u64 << 52) - 1)).max(1);
        let sub = f64::from_bits(mantissa);
        assert_eq!(sqrt_f64(sub).to_bits(), libm::sqrt(sub).to_bits());
    }
}

#[test]
fn sqrt_f32_and_hypot_preserve_extreme_ieee_precision() {
    let mut seed = 0x517C_C1B7_2722_0A95_u64;
    for _ in 0..25_000 {
        let raw = splitmix64_next(&mut seed) as u32;
        let exp = ((raw >> 23) % 254) + 1;
        let mant = raw & ((1_u32 << 23) - 1);
        let xf = f32::from_bits((exp << 23) | mant);
        assert_eq!(
            <f32 as ComplexField>::sqrt(xf).to_bits(),
            sqrt_f32(xf).to_bits()
        );
    }

    let big = hypot_f64(3.0e300, 4.0e300);
    assert!((big - 5.0e300).abs() / 5.0e300 < 1e-15);
    let tiny = hypot_f64(3.0e-305, 4.0e-305);
    assert!((tiny - 5.0e-305).abs() / 5.0e-305 < 1e-15);
    let big32 = hypot_f32(3.0e35, 4.0e35);
    assert!((big32 - 5.0e35).abs() / 5.0e35 < 1e-6);
}

#[test]
fn dual_numbers_propagate_exact_first_derivatives() {
    let x0 = 0.42_f64;
    let x = Dual::variable(x0);
    let h = 1e-7_f64;

    type DualCase = (&'static str, fn(Dual<f64>) -> Dual<f64>, fn(f64) -> f64);
    let cases: [DualCase; 9] = [
        (
            "poly_trig",
            |u| (u * u + Dual::from_f64(1.5)).sin(),
            |v| (v * v + 1.5).sin(),
        ),
        (
            "exp_div",
            |u| u.exp() / (u + Dual::from_f64(2.0)),
            |v| v.exp() / (v + 2.0),
        ),
        (
            "ln_sqrt",
            |u| (u + Dual::one()).sqrt().ln(),
            |v| (v + 1.0).sqrt().ln(),
        ),
        ("tan", |u| u.tan(), |v| v.tan()),
        ("tanh", |u| u.tanh(), |v| v.tanh()),
        ("asin", |u| u.asin(), |v| v.asin()),
        ("acos", |u| u.acos(), |v| v.acos()),
        ("atan", |u| u.atan(), |v| v.atan()),
        ("powf", |u| u.powf(2.75), |v| v.powf(2.75)),
    ];

    for (name, dual_fn, real_fn) in cases {
        let d = dual_fn(x);
        let fd = (real_fn(x0 + h) - real_fn(x0 - h)) / (2.0 * h);
        assert!(
            (d.re - real_fn(x0)).abs() < 1e-14,
            "{name} primal mismatch: {} vs {}",
            d.re,
            real_fn(x0)
        );
        assert!(
            (d.eps - fd).abs() < 1e-8,
            "{name} derivative mismatch: {} vs {fd}",
            d.eps
        );
    }
}

#[test]
fn complex_and_non_copy_borrowed_ring_ops_work_generically() {
    fn eval_quadratic<T: Ring>(a: &T, b: &T, c: &T, x: &T) -> T {
        let ax = a.ref_mul(x);
        let ax_b = ax.ref_add(b);
        ax_b.ref_mul(x).ref_add(c)
    }

    assert_eq!(eval_quadratic(&2_i64, &-3_i64, &5_i64, &4_i64), 25_i64);
    assert_eq!(
        eval_quadratic(&2_i128, &-3_i128, &5_i128, &10_i128),
        175_i128
    );

    let z = Complex64::new(1.5, -0.75);
    let inv = Field::try_inv(&z).expect("nonzero complex invertible");
    let prod = z.ref_mul(&inv);
    assert!((prod.re - 1.0).abs() < 1e-15 && prod.im.abs() < 1e-15);

    let z32 = Complex32::new(0.3, 0.4);
    let (s, c) = ComplexField::sin_cos(z32);
    let unit = s * s + c * c;
    assert!((unit.re - 1.0).abs() < 1e-6 && unit.im.abs() < 1e-6);

    // Overflow-safe Smith division near f64::MAX
    let huge = Complex64::new(1.0e305, -2.0e305);
    let one = Field::ref_div(&huge, &huge);
    assert!((one.re - 1.0).abs() < 1e-14 && one.im.abs() < 1e-14);
}

#[test]
fn workspace_and_primitives_execute_with_zero_heap_allocations() {
    fn assert_scalar<T: Scalar>(_: &T) {}
    assert_scalar(&Dual::new(1.0_f64, 2.0_f64));

    assert_eq!(flatten_2d_index(4, 2, 10), 24);
    assert_eq!(flatten_3d_index(3, 2, 1, 8, 6), 67);
    assert_eq!(*GridIndex(7), 7);
    assert_eq!(*Dimension(16), 16);
    assert!(StepSize::try_new(0.25_f64).is_ok());
    assert!(matches!(
        StepSize::try_new(0.0_f32),
        Err(ScalarError::InvalidStepSize { .. })
    ));

    let mut raw = [1.0_f64; 16];
    let (rem, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        let mut ws = ScalarWorkspace::new(&mut raw);
        ws.zero_fill();
        let a = ws.try_take_slice(6).expect("slice a");
        a[0] = 3.5;
        let (b, mut rest) = ws.try_split(4).expect("split b");
        b[1] = -2.0;
        {
            let mut sub_arena = rest.reborrow();
            let sub_slice = sub_arena.alloc_slice::<f64>(2).expect("sub arena slice");
            assert_eq!(sub_slice.len(), 2);
        }
        let len = rest.len();
        let err = rest.try_take_slice(7).unwrap_err();
        assert_eq!(
            err,
            ScalarError::WorkspaceTooSmall {
                required: 7,
                available: 6
            }
        );
        len
    });
    assert_eq!(rem, 6);
    assert_eq!(raw[0], 3.5);
    assert_eq!(raw[7], -2.0);
}

#[cfg(feature = "serde")]
#[test]
fn scalar_types_and_errors_roundtrip_serde_json() {
    let dual = Dual::new(1.25_f64, -0.5_f64);
    let dual_json = serde_json::to_string(&dual).unwrap();
    assert_eq!(serde_json::from_str::<Dual<f64>>(&dual_json).unwrap(), dual);

    let step = StepSize(0.0625_f64);
    let step_json = serde_json::to_string(&step).unwrap();
    assert_eq!(serde_json::from_str::<StepSize>(&step_json).unwrap(), step);

    let err = ScalarError::WorkspaceTooSmall {
        required: 32,
        available: 8,
    };
    let err_json = serde_json::to_string(&err).unwrap();
    assert_eq!(serde_json::from_str::<ScalarError>(&err_json).unwrap(), err);
}

#[test]
fn point_series_supports_f32_and_f64_with_zero_allocations() {
    use crate::{PointSeries, RealField};

    fn sum_y<T: RealField, P: PointSeries<T> + ?Sized>(pts: &P) -> T {
        let mut acc = T::zero();
        for i in 0..pts.len() {
            acc += pts.y_at(i);
        }
        acc
    }

    let pts32 = [[0.0_f32, 1.5_f32], [1.0, 1.5], [2.0, 1.5], [3.0, 4.0]];
    let pts64 = [(0.0_f64, 2.0_f64), (1.0, 2.0), (2.0, 5.0)];
    let (s32, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        let end32 = PointSeries::advance_plateau_run(&pts32[..], 0, 1.5_f32, 0.0_f32, 10.0_f32);
        assert_eq!(end32, 3);
        let end64 = PointSeries::advance_plateau_run(&pts64[..], 0, 2.0_f64, 0.0_f64, 10.0_f64);
        assert_eq!(end64, 2);
        assert!((sum_y(&pts64) - 9.0_f64).abs() < 1e-12);
        sum_y(&pts32)
    });
    assert!((s32 - 8.5_f32).abs() < 1e-6);
}

#[test]
fn real_field_open_unit_endpoints_and_f32_upper_half_bijection() {
    use crate::RealField;

    let min64 = <f64 as RealField>::open_unit_from_f64(0.0);
    let max64 = <f64 as RealField>::open_unit_from_f64(1.0);
    assert_eq!(min64, 2.0_f64.powi(-54));
    assert_eq!(max64, 1.0 - 2.0_f64.powi(-53));
    assert_eq!(<f64 as RealField>::uniform_open_from_u64(0), min64);
    assert_eq!(<f64 as RealField>::uniform_open_from_u64(u64::MAX), max64);

    let min32 = <f32 as RealField>::open_unit_from_f64(0.0);
    let max32 = <f32 as RealField>::open_unit_from_f64(1.0);
    assert_eq!(min32, 2.0_f32.powi(-25));
    assert_eq!(max32.to_bits(), 0x3f7f_ffff);
    assert_eq!(<f32 as RealField>::uniform_open_from_u64(0), min32);
    assert_eq!(
        <f32 as RealField>::uniform_open_from_u64(1_u64 << 40),
        2.0_f32.powi(-24)
    );
    assert_eq!(<f32 as RealField>::uniform_open_from_u64(u64::MAX), max32);

    let half_even = <f32 as RealField>::uniform_open_from_u64((1_u64 << 23) << 40);
    let half_odd = <f32 as RealField>::uniform_open_from_u64(((1_u64 << 23) + 1) << 40);
    assert!(half_even == 0.5_f32 && half_odd > half_even && half_odd < 1.0_f32);
}
