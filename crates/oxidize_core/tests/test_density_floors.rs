//! Verifies typed `DensityFloorsError` variants and Display formatting in `oxidize_core::traceability`.

#![cfg(feature = "ast-analysis")]

use oxidize_core::traceability::{DensityFloors, DensityFloorsError, parse_density_floors_toml};

#[test]
fn parse_density_floors_toml_returns_typed_errors_and_readable_display() {
    assert_eq!(
        parse_density_floors_toml("[links]\na = \"b.tex\"\n"),
        Ok(None)
    );
    assert_eq!(
        parse_density_floors_toml(
            "[verification_debt]\nmin_verified_density = 0.8\nmin_total_density = 0.5\n"
        ),
        Ok(Some(DensityFloors {
            min_verified_density: 0.8,
            min_total_density: 0.5,
            min_unverified_density: None,
        }))
    );

    let syntax_err = parse_density_floors_toml("not = [valid toml").unwrap_err();
    assert!(matches!(syntax_err, DensityFloorsError::InvalidToml(_)));
    assert!(!syntax_err.to_string().contains('{'));

    let table_err = parse_density_floors_toml("verification_debt = 42\n").unwrap_err();
    assert_eq!(table_err, DensityFloorsError::InvalidVerificationDebtTable);
    assert_eq!(
        table_err.to_string(),
        "[verification_debt] must be a TOML table"
    );

    let neg_err = parse_density_floors_toml(
        "[verification_debt]\nmin_verified_density = -0.1\nmin_unverified_density = 0.2\n",
    )
    .unwrap_err();
    assert_eq!(
        neg_err,
        DensityFloorsError::InvalidDensityValue {
            min_verified_density: -0.1,
            min_total_density: 0.0,
            min_unverified_density: Some(0.2),
        }
    );
    let msg = neg_err.to_string();
    assert!(msg.contains("verified=-0.1") && msg.contains("unverified=0.2"));
    assert!(!msg.contains("Some(") && !msg.contains('{'));
}
