use crate::error::EpidemiologyError;

/// Validates that a population count is strictly positive.
#[verified_engine::verified]
pub fn validate_population(n: f64) -> Result<(), EpidemiologyError> {
    if n <= 0.0 {
        return Err(EpidemiologyError::InvalidParameter {
            name: "n (population)".to_string(),
            value: n,
        });
    }
    Ok(())
}

/// Validates that the initial infected count is within [0, n].
#[verified_engine::verified]
pub fn validate_initial_infected(i0: f64, n: f64) -> Result<(), EpidemiologyError> {
    if i0 < 0.0 || i0 > n {
        return Err(EpidemiologyError::InvalidParameter {
            name: "i0 (initial infected)".to_string(),
            value: i0,
        });
    }
    Ok(())
}

/// Validates that a rate parameter is non-negative.
#[verified_engine::verified]
pub fn validate_rate(name: &str, value: f64) -> Result<(), EpidemiologyError> {
    if value < 0.0 {
        return Err(EpidemiologyError::InvalidParameter {
            name: name.to_string(),
            value,
        });
    }
    Ok(())
}

/// Calculates the Basic Reproduction Number (R0).
///
/// $R_0 = \beta / \gamma$
#[verified_engine::verified]
pub fn basic_reproduction_number(beta: f64, gamma: f64) -> f64 {
    if gamma == 0.0 {
        f64::INFINITY
    } else {
        beta / gamma
    }
}
