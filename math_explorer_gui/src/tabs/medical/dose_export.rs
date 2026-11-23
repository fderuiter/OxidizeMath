//! Scalar-field Marching Squares contour and 16-bit texture export helper for [`super::dose::DoseCalculationTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::dose::DoseCalculationTool;
use super::imrt::{IMRT_GRID_DIM, IMRT_PHANTOM_SIZE_CM};
use math_commons::math_kernel::marching_squares::ScalarField2D;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::medical::dose::kernel::{DoseKernel, ExponentialKernel};

impl DoseCalculationTool {
    /// Evaluates the continuous 2D dose scalar field (4-beam IMRT or point-source PSF) at `(width, height)`.
    pub(crate) fn compute_dose_scalar_field(&self, width: usize, height: usize) -> ScalarField2D {
        let width = width.max(2);
        let height = height.max(2);
        if self.imrt_mode {
            return self.sample_imrt_scalar_field(width, height);
        }

        let kernel = ExponentialKernel::new(self.amplitude, self.beta);
        let dx = IMRT_PHANTOM_SIZE_CM / width as f64;
        let dy = IMRT_PHANTOM_SIZE_CM / height as f64;
        let max_dose = kernel.value_at(0.1).unwrap_or(10.0);
        let mut values = vec![0.0_f64; width * height];

        for y in 0..height {
            let py = (y as f64 - height as f64 / 2.0) * dy;
            for x in 0..width {
                let px = (x as f64 - width as f64 / 2.0) * dx;
                let r = (px * px + py * py).sqrt();
                let dose = if r < 1e-3 {
                    max_dose
                } else {
                    kernel.value_at(r).unwrap_or(0.0).min(max_dose)
                };
                values[flatten_2d_index(x, y, width)] = (1.0 + dose).ln();
            }
        }
        ScalarField2D {
            width,
            height,
            values,
            periodic: false,
            default_levels: 8,
        }
    }

    fn sample_imrt_scalar_field(&self, width: usize, height: usize) -> ScalarField2D {
        let mut src = vec![0.0_f64; IMRT_GRID_DIM * IMRT_GRID_DIM];
        for y in 0..IMRT_GRID_DIM {
            let img_y = IMRT_GRID_DIM - 1 - y;
            for x in 0..IMRT_GRID_DIM {
                let d = self.imrt_workspace.dose_vec[flatten_2d_index(x, y, IMRT_GRID_DIM)];
                src[flatten_2d_index(x, img_y, IMRT_GRID_DIM)] = if d.is_finite() { d } else { 0.0 };
            }
        }
        let values = if width == IMRT_GRID_DIM && height == IMRT_GRID_DIM {
            src
        } else {
            egui_plot::resample_scalar_field_borderless(
                &src,
                IMRT_GRID_DIM,
                IMRT_GRID_DIM,
                width,
                height,
                false,
            )
            .unwrap_or_else(|_| vec![0.0_f64; width * height])
        };
        ScalarField2D {
            width,
            height,
            values,
            periodic: false,
            default_levels: 8,
        }
    }
}
