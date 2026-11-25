// [cite:math_commons]
use crate::prelude::*;
use core::mem::size_of;

#[test]
fn stateless_context_is_zero_sized_and_matches_direct_arithmetic() {
    assert_eq!(size_of::<StatelessContext<f32>>(), 0);
    assert_eq!(size_of::<StatelessContext<f64>>(), 0);
    assert_eq!(size_of::<StatelessContext<Complex64>>(), 0);

    let ctx = StatelessContext::<f64>::new();
    let coeffs = [1.0_f64, -3.0, 2.0, 5.0];
    let x = 2.5_f64;
    let horner_ctx = ctx.eval_horner(&coeffs, &x);
    let horner_direct = ((5.0 * x + 2.0) * x - 3.0) * x + 1.0;
    assert_eq!(horner_ctx.to_bits(), horner_direct.to_bits());
    assert_eq!(ctx.try_div(&6.0, &2.0), Ok(3.0));
    assert_eq!(ctx.try_inv(&0.0), Err(ScalarDomainError::DivisionByZero));
}

#[test]
fn real_field_sqrt_and_hypot_are_zero_ulp_across_200k_normals_and_4k_predecessors() {
    let pred_four = f64::from_bits(4.0_f64.to_bits() - 1);
    assert_eq!(pred_four, 3.9999999999999996);
    assert_eq!(
        <f64 as RealField>::sqrt(pred_four).to_bits(),
        1.9999999999999998_f64.to_bits()
    );
    assert_eq!(
        <f64 as RealField>::sqrt(pred_four).to_bits(),
        libm::sqrt(pred_four).to_bits()
    );

    for k in -500..=500 {
        let pow4 = libm::ldexp(1.0, 2 * k);
        if pow4.is_normal() {
            let pred = f64::from_bits(pow4.to_bits() - 1);
            let succ = f64::from_bits(pow4.to_bits() + 1);
            assert_eq!(
                <f64 as RealField>::sqrt(pred).to_bits(),
                libm::sqrt(pred).to_bits()
            );
            assert_eq!(
                <f64 as RealField>::sqrt(pow4).to_bits(),
                libm::sqrt(pow4).to_bits()
            );
            assert_eq!(
                <f64 as RealField>::sqrt(succ).to_bits(),
                libm::sqrt(succ).to_bits()
            );
        }
    }

    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    for _ in 0..200_000 {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let raw = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let exp_bits = ((raw >> 52) & 0x7FF).clamp(1, 0x7FE);
        let mant_bits = raw & 0x000F_FFFF_FFFF_FFFF;
        let x = f64::from_bits((exp_bits << 52) | mant_bits);
        let got = <f64 as RealField>::sqrt(x);
        let expected = libm::sqrt(x);
        assert_eq!(got.to_bits(), expected.to_bits());
    }

    let big = 1.0e300_f64;
    let h = <f64 as RealField>::hypot(3.0 * big, 4.0 * big);
    assert!(h.is_finite() && (h - 5.0 * big).abs() <= 1e285);
}

#[test]
fn complex_and_dual_scalars_satisfy_field_and_analytic_identities() {
    let z = Complex64::new(0.7, -1.2);
    let inv = z.checked_inverse().expect("nonzero complex inverse");
    let prod = z * inv;
    assert!((prod.re - 1.0).abs() < 1e-15 && prod.im.abs() < 1e-15);

    let root = z.csqrt();
    let sq = root * root;
    assert!((sq.re - z.re).abs() < 1e-14 && (sq.im - z.im).abs() < 1e-14);

    let z32 = Complex32::new(3.0, 4.0);
    assert!((z32.abs() - 5.0).abs() < 1e-6);

    let x = Dual::variable(1.25_f64);
    let f = (x.sin() * x.exp()) / (x * x + Dual::ONE).sqrt();
    let v = 1.25_f64;
    let exact_val = (libm::sin(v) * libm::exp(v)) / libm::sqrt(v * v + 1.0);
    let exact_deriv = libm::exp(v)
        * ((libm::cos(v) + libm::sin(v)) * (v * v + 1.0) - v * libm::sin(v))
        / libm::pow(v * v + 1.0, 1.5);
    assert!(
        (f.re - exact_val).abs() < 1e-15,
        "primal {} vs {}",
        f.re,
        exact_val
    );
    assert!(
        (f.eps - exact_deriv).abs() < 1e-14,
        "tangent {} vs exact {}",
        f.eps,
        exact_deriv
    );
}

#[test]
fn mod_prime_context_evaluates_runtime_gf_p_without_overflow_or_std() {
    assert_eq!(
        ModPrimeContext::try_new(1),
        Err(ScalarDomainError::InvalidModulus { modulus: 1 })
    );

    let prime = 4_294_967_311_u64; // > 2^32 runtime prime requiring u128 intermediate product
    let gf = ModPrimeContext::try_new(prime).expect("valid prime modulus");
    let a = prime - 5;
    let b = prime - 7;
    assert_eq!(gf.mul(&a, &b), 35);

    let inv_a = gf.try_inv(&a).expect("invertible in GF(p)");
    assert_eq!(gf.mul(&a, &inv_a), 1);
    assert_eq!(gf.try_inv(&0), Err(ScalarDomainError::DivisionByZero));

    // Evaluate P(x) = 3 + 5x + 11x^2 (mod 101) at x = 17
    let gf101 = ModPrimeContext::new(101);
    let val = gf101.eval_horner(&[3, 5, 11], &17);
    assert_eq!(val, (3 + 5 * 17 + 11 * 17 * 17) % 101);

    let composite = ModPrimeContext::new(15);
    assert_eq!(
        composite.try_inv(&6),
        Err(ScalarDomainError::NonInvertibleResidue {
            residue: 6,
            modulus: 15,
        })
    );
}

#[cfg(feature = "alloc")]
#[test]
fn dashu_float_and_mod_contexts_operate_without_thread_locals() {
    use dashu::base::Abs;
    use dashu::integer::IBig;
    use dashu::rational::RBig;

    let r1 = RBig::from(3) / RBig::from(7);
    let r1_inv = Field::checked_inverse(&r1).expect("rational inverse");
    assert_eq!(r1 * r1_inv, RBig::one());

    let fctx = DashuFloatContext::try_new(128).expect("128-bit precision");
    let two = fctx.from_i64(2);
    let root2 = fctx.try_sqrt(&two).expect("sqrt(2)");
    let diff = fctx.sub(&fctx.mul(&root2, &root2), &two);
    let tiny = fctx.try_from_f64(1e-35).expect("1e-35");
    assert!(diff.clone().abs() < tiny);

    let p = IBig::from(1_000_000_000_039_u64);
    let mctx = DashuModContext::try_new(p).expect("prime modulus");
    let elem = IBig::from(123_456_789_u64);
    let inv = mctx.try_inv(&elem).expect("modular inverse");
    assert_eq!(mctx.mul(&elem, &inv), IBig::from(1u8));
    assert_eq!(mctx.add(&elem, &mctx.neg(&elem)), IBig::from(0u8));
    assert_eq!(
        mctx.try_inv(&IBig::from(0u8)),
        Err(ScalarDomainError::DivisionByZero)
    );
    assert!(DashuFloatContext::try_new(0).is_err());
    assert!(DashuModContext::try_new(IBig::from(1u8)).is_err());
}

#[test]
fn primitive_integer_ring_identities() {
    assert_eq!(<i32 as Ring>::zero(), 0);
    assert_eq!(<i32 as Ring>::one(), 1);
    assert!(<i32 as Ring>::zero().is_zero());
    assert!(!<i32 as Ring>::one().is_zero());
    assert_eq!(3_i32.pow_u64(0), 1);
    assert_eq!(3_i32.pow_u64(5), 243);
    assert_eq!((-2_i64).pow_u64(9), -512);
    assert_eq!(5_i128.pow_u64(6), 15_625);
    assert_eq!(7_isize.pow_u64(3), 343);
}

#[test]
fn stateless_integer_horner_identities() {
    let ctx = StatelessContext::<i64>::default();
    assert_eq!(ctx.zero(), 0);
    assert_eq!(ctx.one(), 1);
    assert!(ctx.is_zero(&0));
    assert_eq!(ctx.add(&9, &-4), 5);
    assert_eq!(ctx.sub(&9, &4), 5);
    assert_eq!(ctx.mul(&-6, &7), -42);
    assert_eq!(ctx.neg(&13), -13);
    assert_eq!(ctx.pow_u64(&2, 10), 1024);
    assert_eq!(ctx.eval_horner(&[], &99), 0);
    assert_eq!(ctx.eval_horner(&[7, -2, 3], &4), 7 - 8 + 48);
}

#[test]
fn f32_and_f64_real_field_algebraic_and_rounding_identities() {
    assert_eq!(<f32 as Field>::checked_inverse(&0.0), None);
    assert_eq!(<f64 as Field>::checked_inverse(&0.0), None);
    assert_eq!(<f32 as Field>::checked_div(&8.0, &2.0), Some(4.0));
    assert_eq!(<f64 as Field>::checked_div(&8.0, &0.0), None);
    assert!((<f32 as ComplexField>::cbrt(27.0) - 3.0).abs() < 1e-6);
    assert!((<f64 as ComplexField>::cbrt(-27.0) + 3.0).abs() < 1e-14);
    assert_eq!(<f64 as RealField>::copysign(3.5, -1.0), -3.5);
    assert_eq!(<f64 as RealField>::clamp(5.0, -1.0, 2.0), 2.0);
    assert_eq!(<f64 as RealField>::min(3.0, -2.0), -2.0);
    assert_eq!(<f64 as RealField>::max(3.0, -2.0), 3.0);
    assert!((<f64 as RealField>::rem_euclid(-1.25, 1.0) - 0.75).abs() < 1e-15);
    assert_eq!(<f64 as RealField>::from_usize(42), 42.0);
    assert_eq!(<f64 as RealField>::from_i32(-17), -17.0);
    assert_eq!(<f32 as RealField>::to_f64(2.5_f32), 2.5_f64);
}

#[test]
fn f64_real_and_complex_field_transcendental_identities() {
    assert!((<f64 as ComplexField>::log2(8.0) - 3.0).abs() < 1e-15);
    assert!((<f64 as ComplexField>::log10(1000.0) - 3.0).abs() < 1e-15);
    assert!((<f64 as RealField>::expm1(1e-10) - 1e-10).abs() < 1e-20);
    assert!((<f64 as RealField>::ln_1p(1e-10) - 1e-10).abs() < 1e-20);
    let (s, c) = <f64 as ComplexField>::sin_cos(0.7);
    assert!((s * s + c * c - 1.0).abs() < 1e-15);
    let ch = <f64 as ComplexField>::cosh(0.8);
    let sh = <f64 as ComplexField>::sinh(0.8);
    assert!((ch * ch - sh * sh - 1.0).abs() < 1e-14);
    assert!(
        (<f64 as ComplexField>::asin(0.5) + <f64 as ComplexField>::acos(0.5)
            - <f64 as RealField>::FRAC_PI_2)
            .abs()
            < 1e-15
    );
    assert!((<f64 as ComplexField>::atan(1.0) * 4.0 - <f64 as RealField>::PI).abs() < 1e-15);
    assert!((<f64 as ComplexField>::asinh(1.2).sinh() - 1.2).abs() < 1e-14);
    assert!((<f64 as ComplexField>::acosh(2.2).cosh() - 2.2).abs() < 1e-14);
    assert!((<f64 as ComplexField>::atanh(0.4).tanh() - 0.4).abs() < 1e-14);
    assert_eq!(<f64 as ComplexField>::floor(2.9), 2.0);
    assert_eq!(<f64 as ComplexField>::ceil(2.1), 3.0);
    assert_eq!(<f64 as ComplexField>::round(-2.6), -3.0);
    assert_eq!(<f64 as ComplexField>::signum(-9.0), -1.0);
}

#[test]
fn dual_and_complex_transcendentals_and_error_diagnostics() {
    let x = Dual::variable(0.6_f64);
    let c = Dual::constant(2.0_f64);
    assert_eq!(c.re, 2.0);
    assert_eq!(c.eps, 0.0);
    assert!((x.cos().eps + 0.6_f64.sin()).abs() < 1e-15);
    assert!((x.tan().eps - 1.0 / 0.6_f64.cos().powi(2)).abs() < 1e-14);
    assert!((x.ln().eps - 1.0 / 0.6_f64).abs() < 1e-15);
    assert!((x.cbrt().eps - 1.0 / (3.0 * 0.6_f64.cbrt().powi(2))).abs() < 1e-14);
    assert!((x.asin().eps - 1.0 / (1.0 - 0.36_f64).sqrt()).abs() < 1e-15);
    assert!((x.acos().eps + 1.0 / (1.0 - 0.36_f64).sqrt()).abs() < 1e-15);
    assert!((x.atan().eps - 1.0 / (1.0 + 0.36_f64)).abs() < 1e-15);
    assert!((x.sinh().eps - 0.6_f64.cosh()).abs() < 1e-15);
    assert!((x.cosh().eps - 0.6_f64.sinh()).abs() < 1e-15);
    assert!((x.tanh().eps - 1.0 / 0.6_f64.cosh().powi(2)).abs() < 1e-15);
    assert_eq!(Dual::constant(0.0_f64).checked_inverse(), None);

    let i = <Complex64 as ComplexField>::imag_unit();
    assert_eq!(i.re, 0.0);
    assert_eq!(i.im, 1.0);
    assert_eq!(i.conj(), Complex64::new(0.0, -1.0));
    assert_eq!(Complex64::ZERO.checked_inverse(), None);
    let euler = (i * Complex64::from_real(core::f64::consts::PI)).exp();
    assert!((euler.re + 1.0).abs() < 1e-15 && euler.im.abs() < 1e-15);
    let z = Complex64::new(0.4, 0.3);
    assert!(((z.sin() * z.sin() + z.cos() * z.cos()).re - 1.0).abs() < 1e-14);
    assert!(((z.ln().exp() - z).abs()) < 1e-14);
}

#[test]
fn scalar_domain_error_codes_and_diagnostic_visitor() {
    use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};

    struct CountVisitor(usize);
    impl DiagnosticVisitor for CountVisitor {
        fn record_str(&mut self, _field: &str, _value: &str) {
            self.0 += 1;
        }
        fn record_u64(&mut self, _field: &str, _value: u64) {
            self.0 += 1;
        }
        fn record_usize(&mut self, _field: &str, _value: usize) {
            self.0 += 1;
        }
        fn record_display(&mut self, _field: &str, _value: &dyn core::fmt::Display) {
            self.0 += 1;
        }
    }

    let e0 = ScalarDomainError::DivisionByZero;
    let e1 = ScalarDomainError::InvalidModulus { modulus: 0 };
    let e2 = ScalarDomainError::NonInvertibleResidue {
        residue: 4,
        modulus: 12,
    };
    let e3 = ScalarDomainError::InvalidPrecision { precision_bits: 0 };
    assert_eq!(e0.error_type(), "ScalarDomainError");
    assert_eq!(e1.code(), "SCALAR-001");
    assert_eq!(e0.code(), "SCALAR-002");
    assert_eq!(e2.code(), "SCALAR-003");
    assert_eq!(e3.code(), "SCALAR-004");
    assert_eq!(e0.severity(), Severity::Error);

    let mut visitor = CountVisitor(0);
    e0.visit_fields(&mut visitor);
    e1.visit_fields(&mut visitor);
    e2.visit_fields(&mut visitor);
    e3.visit_fields(&mut visitor);
    assert_eq!(visitor.0, 16);
}
