use crate::error::EngineeringError;

/// A baud rate in symbols per second.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BaudRate(u32);

impl BaudRate {
    /// Creates a new `BaudRate`.
    #[verified_engine::verified]
    pub fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the inner value.
    #[verified_engine::verified]
    pub fn value(&self) -> u32 {
        self.0
    }
}

/// The number of payload data bits in a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DataBits(u32);

impl DataBits {
    /// Creates a new `DataBits`.
    #[verified_engine::verified]
    pub fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the inner value.
    #[verified_engine::verified]
    pub fn value(&self) -> u32 {
        self.0
    }
}

/// The total number of bits in a frame (start, data, parity, stop).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct TotalBits(u32);

impl TotalBits {
    /// Creates a new `TotalBits`.
    ///
    /// # Errors
    /// Returns `EngineeringError` if `value` is 0, since a frame cannot have 0 bits.
    #[verified_engine::verified]
    pub fn new(value: u32) -> Result<Self, EngineeringError> {
        if value == 0 {
            return Err(EngineeringError::InvalidParameter {
                name: "TotalBits".to_string(),
                value: 0.0,
            });
        }
        Ok(Self(value))
    }

    /// Returns the inner value.
    #[verified_engine::verified]
    pub fn value(&self) -> u32 {
        self.0
    }
}

/// Calculates the Effective UART Throughput using type-safe inputs.
///
/// In real-world communications, the "Baud Rate" is not the actual data transfer rate.
/// Overhead from start bits, stop bits, and parity bits reduces the effective throughput.
///
/// $$ \text{Eff. Speed} = \frac{\text{Baud Rate} \times \text{Data Bits}}{\text{Total Bits per Frame}} $$
///
/// # Arguments
///
/// * `baud_rate` - [`BaudRate`]
/// * `data_bits` - [`DataBits`]
/// * `total_bits` - [`TotalBits`]
///
/// # Returns
///
/// * `f64` - Effective bits per second.
///
/// # Example
///
/// ```
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::engineering::{calculate_uart_throughput, BaudRate, DataBits, TotalBits};
///
/// // Standard 9600 8N1
/// let throughput = calculate_uart_throughput(
///     BaudRate::new(9600),
///     DataBits::new(8),
///     TotalBits::new(10).unwrap(),
/// );
/// assert_eq!(throughput, 7680.0);
/// ```
#[verified_engine::verified]
pub fn calculate_uart_throughput(
    baud_rate: BaudRate,
    data_bits: DataBits,
    total_bits: TotalBits,
) -> f64 {
    (baud_rate.value() as f64 * data_bits.value() as f64) / total_bits.value() as f64
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for TotalBits {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = u32::deserialize(deserializer)?;
        Self::new(raw).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_calculate_uart_throughput() {
        let eff = calculate_uart_throughput(
            BaudRate::new(9600),
            DataBits::new(8),
            TotalBits::new(10).unwrap(),
        );
        assert_eq!(eff, 7680.0);
    }
}
