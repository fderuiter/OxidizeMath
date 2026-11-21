use crate::error::EngineeringError;

/// The cutting speed ($V$) in meters per minute (m/min).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct CuttingSpeed(f64);

impl CuttingSpeed {
    /// Creates a new `CuttingSpeed`.
    ///
    /// # Errors
    /// Returns `EngineeringError` if `value` is negative or zero.
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, EngineeringError> {
        if value <= 0.0 {
            return Err(EngineeringError::InvalidParameter {
                name: "CuttingSpeed".to_string(),
                value,
            });
        }
        Ok(Self(value))
    }

    /// Returns the inner value.
    #[verified_engine::verified]
    pub fn value(&self) -> f64 {
        self.0
    }
}

/// A material-specific constant ($C$) for Taylor's equation.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct MaterialConstant(f64);

impl MaterialConstant {
    /// Creates a new `MaterialConstant`.
    ///
    /// # Errors
    /// Returns `EngineeringError` if `value` is negative.
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, EngineeringError> {
        if value < 0.0 {
            return Err(EngineeringError::InvalidParameter {
                name: "MaterialConstant".to_string(),
                value,
            });
        }
        Ok(Self(value))
    }

    /// Returns the inner value.
    #[verified_engine::verified]
    pub fn value(&self) -> f64 {
        self.0
    }
}

/// The exponent ($n$) dependent on tool material for Taylor's equation.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ToolExponent(f64);

impl ToolExponent {
    /// Creates a new `ToolExponent`.
    ///
    /// # Errors
    /// Returns `EngineeringError` if `value` is zero, to prevent division by zero in the exponent.
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, EngineeringError> {
        if value == 0.0 {
            return Err(EngineeringError::InvalidParameter {
                name: "ToolExponent".to_string(),
                value,
            });
        }
        Ok(Self(value))
    }

    /// Returns the inner value.
    #[verified_engine::verified]
    pub fn value(&self) -> f64 {
        self.0
    }
}

/// The estimated tool life in minutes.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ToolLife(f64);

impl ToolLife {
    /// Creates a new `ToolLife`.
    #[verified_engine::verified]
    pub fn new(value: f64) -> Self {
        Self(value)
    }

    /// Returns the inner value.
    #[verified_engine::verified]
    pub fn value(&self) -> f64 {
        self.0
    }
}

/// Estimates Fault Lifetime using a Modified Taylor's Equation and type-safe inputs.
///
/// Used in predictive maintenance (Prognostics) for CNC machines.
/// It models the relationship between cutting speed and tool life.
///
/// The standard Taylor's Tool Life Equation is $V T^n = C$.
/// This implementation calculates $T$ (Time) given $V$ (Velocity).
///
/// $$ T = \left( \frac{C}{V} \right)^{1/n} $$
///
/// # Arguments
///
/// * `cutting_speed` ($V$) - [`CuttingSpeed`] in m/min.
/// * `constant_c` ($C$) - [`MaterialConstant`].
/// * `exponent_n` ($n$) - [`ToolExponent`].
///
/// # Returns
///
/// * [`ToolLife`] - Estimated Tool life ($T$) in minutes.
///
/// # Example
///
/// ```
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::engineering::{calculate_taylor_tool_life, CuttingSpeed, MaterialConstant, ToolExponent};
///
/// // High Speed Steel (HSS) tool cutting steel
/// let life = calculate_taylor_tool_life(
///     CuttingSpeed::new(40.0).unwrap(),
///     MaterialConstant::new(100.0).unwrap(),
///     ToolExponent::new(0.1).unwrap(),
/// );
/// // Expecting (100/40)^(1/0.1) = 2.5^10 = 9536.7...
/// assert!(life.value() > 9000.0);
/// ```
#[verified_engine::verified]
pub fn calculate_taylor_tool_life(
    cutting_speed: CuttingSpeed,
    constant_c: MaterialConstant,
    exponent_n: ToolExponent,
) -> ToolLife {
    let result = (constant_c.value() / cutting_speed.value()).powf(1.0 / exponent_n.value());
    ToolLife::new(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_calculate_taylor_tool_life() {
        let life = calculate_taylor_tool_life(
            CuttingSpeed::new(40.0).unwrap(),
            MaterialConstant::new(100.0).unwrap(),
            ToolExponent::new(0.1).unwrap(),
        );
        assert!(life.value() > 9000.0);
    }
}
