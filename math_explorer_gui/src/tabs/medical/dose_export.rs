//! 2D scalar-field export adapter for [`DoseCalculationTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::dose::DoseCalculationTool;
use super::imrt::{IMRT_GRID_DIM, IMRT_PHANTOM_SIZE_CM};
use crate::framework::ScalarField2D;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::medical::dose::kernel::{DoseKernel, ExponentialKernel};

/// Extracts the continuous 2D radiation dose distribution (in Gy / kernel fluence units)
/// aligned top-down (`y = 0` at top row) with [`DoseCalculationTool`]'s viewport image.
pub(super) fn build_dose_scalar_field(tool: &DoseCalculationTool) -> ScalarField2D {
    if tool.imrt_mode {
        let dose_slice = super::dose::canvas::active_imrt_dose_slice(tool);
        let mut values = vec![0.0_f64; IMRT_GRID_DIM * IMRT_GRID_DIM];
        for y in 0..IMRT_GRID_DIM {
            let img_y = IMRT_GRID_DIM - 1 - y;
            for x in 0..IMRT_GRID_DIM {
                let dose = dose_slice[flatten_2d_index(x, y, IMRT_GRID_DIM)];
                values[flatten_2d_index(x, img_y, IMRT_GRID_DIM)] =
                    if dose.is_finite() { dose } else { 0.0 };
            }
        }
        return ScalarField2D::continuous(IMRT_GRID_DIM, IMRT_GRID_DIM, values, false);
    }

    let width = tool.width.max(2);
    let height = tool.height.max(2);
    let mut values = vec![0.0_f64; width * height];
    let kernel = ExponentialKernel::new(tool.amplitude, tool.beta);
    let dx = IMRT_PHANTOM_SIZE_CM / width as f64;
    let dy = IMRT_PHANTOM_SIZE_CM / height as f64;
    let max_dose = kernel.value_at(0.1).unwrap_or(10.0);

    for y in 0..height {
        let py = (y as f64 - height as f64 / 2.0) * dy;
        for x in 0..width {
            let px = (x as f64 - width as f64 / 2.0) * dx;
            let r = (px * px + py * py).sqrt();
            let dose = if r < 1e-3 {
                max_dose
            } else {
                kernel.value_at(r).unwrap_or(0.0)
            };
            values[flatten_2d_index(x, y, width)] = if dose.is_finite() { dose } else { 0.0 };
        }
    }
    ScalarField2D::continuous(width, height, values, false)
}
