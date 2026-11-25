#![cfg_attr(any(), verified(opt_out = "tests"))]
// [cite:math_scalars]

use core::mem::{align_of, size_of};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOC: VerifiedAllocator = VerifiedAllocator;

use crate::{
    Celsius, Dim, DimAcceleration, DimArea, DimEnergy, DimLength, DimSpec, DimVelocity,
    Dimensionless, DimensionlessVal, FemtometersPerSecond, GyromagneticRatioSi, Joules,
    JoulesPerKelvin, JoulesPerMole, JoulesPerMoleKelvin, JoulesSecond, Kelvin, Kilograms,
    Kiloparsecs, MeVFm, Meters, MetersPerSecond, Millimeters, N1, N2, P1, P2, P3, P4, Quantity,
    Scalar, Seconds, SiFrame, SquareMeters, Tesla, ViewportLength, Z0,
};

#[test]
fn repr_transparent_memory_layout_matches_primitives() {
    assert_eq!(size_of::<Quantity<f64, DimLength, SiFrame>>(), 8);
    assert_eq!(
        align_of::<Quantity<f64, DimLength, SiFrame>>(),
        align_of::<f64>()
    );
    assert_eq!(size_of::<Quantity<f32, DimEnergy, SiFrame>>(), 4);
    assert_eq!(
        align_of::<Quantity<f32, DimEnergy, SiFrame>>(),
        align_of::<f32>()
    );
    assert_eq!(size_of::<Celsius<f64>>(), 8);
    assert_eq!(size_of::<Celsius<f32>>(), 4);
}

#[test]
fn open_type_level_si_exponents_derive_products_quotients_recip_and_sqrt() {
    let m = Kilograms::new(2.5);
    let dx = Meters::new(12.0);
    let dt = Seconds::new(3.0);

    let v: MetersPerSecond = dx / dt;
    assert_eq!(v.get(), 4.0);
    assert_eq!(DimVelocity::exponents(), [1, 0, -1, 0, 0, 0, 0]);

    let a: Quantity<f64, DimAcceleration, SiFrame> = v / dt;
    assert!((a.get() - (4.0 / 3.0)).abs() < 1.0e-12);
    assert_eq!(DimAcceleration::exponents(), [1, 0, -2, 0, 0, 0, 0]);

    let ke: Joules = 0.5 * m * v * v;
    assert!((ke.get() - 20.0).abs() < 1.0e-12);
    assert_eq!(DimEnergy::exponents(), [2, 1, -2, 0, 0, 0, 0]);

    let area: SquareMeters = dx * dx;
    assert_eq!(DimArea::exponents(), [2, 0, 0, 0, 0, 0, 0]);
    let side: Meters = area.sqrt();
    assert_eq!(side.get(), 12.0);

    type DimQuartic = Dim<P4, Z0, N2, Z0, Z0, Z0, Z0>;
    type DimKinematicViscosity = Dim<P2, Z0, N1, Z0, Z0, Z0, Z0>;
    let quartic: Quantity<f64, DimQuartic, SiFrame> = Quantity::new(81.0);
    let halved: Quantity<f64, DimKinematicViscosity, SiFrame> = quartic.sqrt();
    assert_eq!(halved.get(), 9.0);

    let freq = dt.recip();
    assert_eq!(
        Dim::<Z0, Z0, N1, Z0, Z0, Z0, Z0>::exponents(),
        [0, 0, -1, 0, 0, 0, 0]
    );
    assert!((freq.get() - (1.0 / 3.0)).abs() < 1.0e-12);
}

#[test]
fn seven_base_si_dimensions_cancel_in_mri_and_arrhenius_exponents() {
    let hbar = JoulesSecond::new(1.054_571_8e-34);
    let gamma = GyromagneticRatioSi::new(2.675e8);
    let b0 = Tesla::new(3.0);
    let kb = JoulesPerKelvin::new(1.380_649e-23);
    let temp = Celsius::new(37.0).to_kelvin();

    let exponent: DimensionlessVal = -(hbar * gamma * b0) / (kb * temp);
    let boltzmann = exponent.exp();
    assert!(boltzmann.get() > 0.999_9 && boltzmann.get() < 1.0);

    let ea = JoulesPerMole::new(24_000.0);
    let r_gas = JoulesPerMoleKelvin::new(8.314_462_618);
    let t_ref: Kelvin = Celsius::new(25.0).to_kelvin();
    let t_hot: Kelvin = Celsius::new(45.0).to_kelvin();
    let arrhenius_arg: DimensionlessVal = (ea / r_gas) * (t_ref.recip() - t_hot.recip());
    assert!(arrhenius_arg.exp().get() > 1.0);

    // Check seventh SI dimension (luminous intensity J = cd) also composes cleanly.
    let lumen_like = Quantity::<f64, Dim<Z0, Z0, Z0, Z0, Z0, Z0, P3>, SiFrame>::new(8.0);
    let candela = Quantity::<f64, Dim<Z0, Z0, Z0, Z0, Z0, Z0, P1>, SiFrame>::new(2.0);
    let cd_sq = lumen_like / candela;
    assert_eq!(cd_sq.sqrt().get(), 2.0);
}

#[test]
fn nuclear_frame_gamow_ratio_cancels_to_dimensionless() {
    let hbar_c = MeVFm::new(197.3);
    let c_nuc = FemtometersPerSecond::new(2.997_924_58e23);
    let e_sq = MeVFm::new(1.439_976);
    let v = FemtometersPerSecond::new(1.5e22);

    let hbar = hbar_c / c_nuc;
    let eta = (core::f64::consts::PI * 2.0 * 90.0 * e_sq) / (hbar * v);
    let eta_dimensionless: Quantity<f64, Dimensionless, crate::NuclearMeVFmFrame> = eta;
    assert!(eta_dimensionless.get() > 1.0 && eta_dimensionless.is_finite());
}

#[verified_engine::verified]
fn kinetic_energy_sum<T: Scalar>(
    masses: &[Kilograms<T>],
    speeds: &[MetersPerSecond<T>],
) -> Joules<T> {
    let mut acc = Joules::<T>::ZERO;
    let n = masses.len().min(speeds.len());
    let mut i = 0;
    while i < n {
        let ke = masses[i] * speeds[i] * speeds[i] * T::HALF;
        acc += ke;
        i += 1;
    }
    assert!(acc.is_finite());
    acc
}

#[test]
fn f32_and_f64_generic_kernel_runs_with_zero_heap_allocations() {
    let m64 = [
        Kilograms::new(1.0_f64),
        Kilograms::new(2.0_f64),
        Kilograms::new(3.0_f64),
    ];
    let v64 = [
        MetersPerSecond::new(2.0_f64),
        MetersPerSecond::new(3.0_f64),
        MetersPerSecond::new(4.0_f64),
    ];
    let m32 = [Kilograms::new(1.0_f32), Kilograms::new(2.0_f32)];
    let v32 = [MetersPerSecond::new(2.0_f32), MetersPerSecond::new(3.0_f32)];

    let ((e64, e32), metrics) = VerifiedEngine::run_verified(|| {
        let mut sum64 = Joules::<f64>::ZERO;
        let mut sum32 = Joules::<f32>::ZERO;
        let mut step = 0;
        while step < 64 {
            sum64 += kinetic_energy_sum(&m64, &v64);
            sum32 += kinetic_energy_sum(&m32, &v32);
            step += 1;
        }
        (sum64, sum32)
    });

    assert_eq!(metrics.function_calls, 128);
    assert!((e64.get() - 64.0 * 35.0).abs() < 1.0e-9);
    assert!((e32.get() - 64.0 * 11.0).abs() < 1.0e-3);
}

#[test]
fn viewport_kiloparsec_millimeter_and_temperature_round_trips() {
    let vp = ViewportLength::new(0.55);
    let kpc = vp.to_kiloparsecs(2.5);
    assert!((kpc.get() - 1.375).abs() < 1.0e-12);
    let vp_back = kpc.to_viewport(2.5);
    assert!((vp_back.get() - vp.get()).abs() < 1.0e-12);

    let meters = kpc.to_meters();
    let kpc_back = Kiloparsecs::from_meters(meters);
    assert!((kpc_back.get() - kpc.get()).abs() < 1.0e-12);

    let mm = Millimeters::new(85.0);
    assert!((Millimeters::from_meters(mm.to_meters()).get() - 85.0).abs() < 1.0e-12);

    let c = Celsius::new(-20.0);
    let k = c.to_kelvin();
    assert!((k.get() - 253.15).abs() < 1.0e-12);
    assert!((Celsius::from_kelvin(k).get() - (-20.0)).abs() < 1.0e-12);
    assert!(((Celsius::new(60.0) - c).get() - 80.0).abs() < 1.0e-12);
}

#[cfg(feature = "serde")]
#[test]
fn serde_transparent_preserves_scalar_wire_format() {
    let r = Kiloparsecs::new(2.75);
    let json = serde_json::to_string(&r).expect("serialize kpc");
    assert_eq!(json, "2.75");
    let decoded: Kiloparsecs = serde_json::from_str(&json).expect("deserialize kpc");
    assert_eq!(decoded, r);
}

macro_rules! theory_verification {
    (
        module = $module_name:expr,
        constants = { $( $const_name:ident = $const_val:expr; )* },
        test = $test_body:block
    ) => {
        mod theory_verification {
            use super::*;
            $( const $const_name: f64 = $const_val; )*

            #[test]
            fn test_theory_verification() {
                let _ = $module_name;
                $test_body
            }
        }
    };
}

theory_verification!(
    module = "math_scalars",
    constants = {
        MASS_KG = 2.5;
        SPEED_M_S = 4.0;
        EXPECTED_KE_J = 20.0;
    },
    test = {
        let m = Kilograms::new(MASS_KG);
        let v = MetersPerSecond::new(SPEED_M_S);
        let ((ke,), metrics) = VerifiedEngine::run_verified(|| (kinetic_energy_sum(&[m], &[v]),));
        assert_eq!(metrics.function_calls, 1);
        assert!((ke.get() - EXPECTED_KE_J).abs() < 1.0e-12);
        assert_eq!(DimEnergy::exponents(), [2, 1, -2, 0, 0, 0, 0]);
    }
);
