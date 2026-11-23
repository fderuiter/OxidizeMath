//! Integration tests for astrophysics.
use domain_physics::physics::astrophysics::galaxies::{
    GalaxyModel, GeneralIrregular, Magnitude, Mpc, SolarMassLog, TypeCode10, TypeCode95To99,
};

const F64_TOLERANCE: f64 = math_commons::registry::TOLERANCE_STANDARD;

#[test]
#[verified_engine::verified]
fn test_calculate_log_mass_from_distance() {
    // Test case for GeneralIrregular
    let distance = Mpc(10.0); // Mpc
    let expected = 0.0230 * distance.as_f64() + 0.7840;
    let result = GeneralIrregular.log_mass_from_distance(distance).as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);

    // Test case for TypeCode10
    let expected = 0.0250 * distance.as_f64() + 7.6860;
    let result = TypeCode10.log_mass_from_distance(distance).as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);

    // Test case for TypeCode95To99
    let expected = 0.0504 * distance.as_f64() + 7.5715;
    let result = TypeCode95To99.log_mass_from_distance(distance).as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);
}

#[test]
#[verified_engine::verified]
fn test_calculate_apparent_magnitude_from_distance() {
    let distance = Mpc(10.0); // Mpc

    // Test case for GeneralIrregular
    let expected = 0.0206 * distance.as_f64() + 16.0010;
    let result = GeneralIrregular
        .apparent_magnitude_from_distance(distance)
        .unwrap()
        .as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);

    // Test case for TypeCode10
    let expected = 0.0140 * distance.as_f64() + 16.575;
    let result = TypeCode10
        .apparent_magnitude_from_distance(distance)
        .unwrap()
        .as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);

    // Test case for TypeCode95To99 (should be None)
    let result = TypeCode95To99.apparent_magnitude_from_distance(distance);
    assert!(result.is_none());
}

#[test]
#[verified_engine::verified]
fn test_calculate_log_mass_from_absolute_magnitude() {
    let absolute_magnitude_v = Magnitude(-15.0);

    // Test case for GeneralIrregular
    let expected = -0.6670 * absolute_magnitude_v.as_f64() - 1.4975;
    let result = GeneralIrregular
        .log_mass_from_absolute_magnitude(absolute_magnitude_v)
        .as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);

    // Test case for TypeCode10
    let result = TypeCode10
        .log_mass_from_absolute_magnitude(absolute_magnitude_v)
        .as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);

    // Test case for TypeCode95To99
    let expected = -0.3837 * absolute_magnitude_v.as_f64() - 2.2864;
    let result = TypeCode95To99
        .log_mass_from_absolute_magnitude(absolute_magnitude_v)
        .as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);
}

#[test]
#[verified_engine::verified]
fn test_calculate_redshift_from_log_mass() {
    let log_mass_solar = SolarMassLog(10.0);

    // Test case for GeneralIrregular
    let expected = 0.0094 * log_mass_solar.as_f64() - 0.7270;
    let result = GeneralIrregular
        .redshift_from_log_mass(log_mass_solar)
        .as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);

    // Test case for TypeCode10
    let expected = 0.00093 * log_mass_solar.as_f64() - 0.0716;
    let result = TypeCode10.redshift_from_log_mass(log_mass_solar).as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);

    // Test case for TypeCode95To99
    let expected = 0.0031 * log_mass_solar.as_f64() - 0.0223;
    let result = TypeCode95To99
        .redshift_from_log_mass(log_mass_solar)
        .as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);
}
