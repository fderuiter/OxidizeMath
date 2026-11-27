//! # Engineering Solvers & Signal Processing
//!
//! Verified models spanning RF/mmWave radar signal processing, coupled electrochemical battery
//! degradation, predictive CNC machining tool life, and asynchronous serial link throughput.

pub mod battery_degradation;
/// Taylor tool-life prognostics for CNC machining (`V * T^n = C`).
pub mod cnc;
pub mod radar;
/// Asynchronous UART frame overhead and effective payload throughput calculators.
pub mod uart;

pub use crate::error;
pub use cnc::*;
pub use uart::*;

// [cite:engineering]

use pure_math::theory_verification;

theory_verification!(
    module = "engineering",
    constants = {},
    test = {
        use approx::assert_relative_eq;

        // Physical Invariant 1: Taylor's invariant V * T^n = C holds across cutting speeds.
        let c = MaterialConstant::new(120.0).unwrap();
        let n = ToolExponent::new(0.25).unwrap();
        let v1 = CuttingSpeed::new(40.0).unwrap();
        let v2 = CuttingSpeed::new(80.0).unwrap();
        let t1 = calculate_taylor_tool_life(v1, c, n).value();
        let t2 = calculate_taylor_tool_life(v2, c, n).value();
        assert_relative_eq!(
            v1.value() * t1.powf(n.value()),
            c.value(),
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            v2.value() * t2.powf(n.value()),
            c.value(),
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        // Doubling speed with n = 0.25 reduces tool life by 2^(1 / 0.25) = 16x.
        assert_relative_eq!(
            t1 / t2,
            16.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );

        // Physical Invariant 2: Asynchronous 8N1 UART frame efficiency equals 8 / 10 = 0.8.
        let baud = BaudRate::new(115_200);
        let eff = calculate_uart_throughput(baud, DataBits::new(8), TotalBits::new(10).unwrap());
        assert_relative_eq!(
            eff / f64::from(baud.value()),
            0.8,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }
);
