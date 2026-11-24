#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::formula::CompiledFractalJob;
use super::runner::{
    clamp_fractal_dimensions, FractalMode, FractalRenderRequest, COORD_MAX, COORD_MIN,
    MAX_FRACTAL_TEXTURE_SIDE, ZOOM_MAX, ZOOM_MIN,
};
use super::FractalViewer;
use math_commons::math_kernel::marching_squares::ScalarField2D;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::chaos::fractals::{smooth_escape_julia, smooth_escape_mandelbrot};
use num_complex::Complex;

/// Evaluates a smooth continuous escape-potential [`ScalarField2D`] with an optional [`CompiledFractalJob`].
#[must_use]
pub fn render_fractal_scalar_field_with_formula(
    req: &FractalRenderRequest,
    formula: Option<&CompiledFractalJob>,
) -> ScalarField2D {
    let (width, height) = clamp_fractal_dimensions(req.width, req.height, MAX_FRACTAL_TEXTURE_SIDE);
    if width == 0 || height == 0 {
        return ScalarField2D {
            width: 1,
            height: 1,
            values: vec![0.0],
            periodic: false,
            default_levels: 10,
        };
    }
    let zoom = if req.zoom.is_finite() && req.zoom >= ZOOM_MIN {
        req.zoom.min(ZOOM_MAX)
    } else {
        1.0
    };
    let scale_y = 3.0 / zoom;
    let scale_x = scale_y * (width as f64 / height as f64);
    let x_start = req.center.re - scale_x * 0.5;
    let y_start = req.center.im + scale_y * 0.5;
    let mut values = vec![0.0_f64; width * height];

    for y in 0..height {
        let im = y_start - (y as f64 / height as f64) * scale_y;
        for x in 0..width {
            let re = x_start + (x as f64 / width as f64) * scale_x;
            let pt = Complex::new(re, im);
            let pot = match req.mode {
                FractalMode::Mandelbrot => smooth_escape_mandelbrot(pt, req.max_iter),
                FractalMode::Julia => smooth_escape_julia(pt, req.julia_c, req.max_iter),
                FractalMode::CustomMandelbrot => formula.map_or_else(
                    || smooth_escape_mandelbrot(pt, req.max_iter),
                    |job| job.smooth_escape(Complex::new(0.0, 0.0), pt, req.max_iter),
                ),
                FractalMode::CustomJulia => formula.map_or_else(
                    || smooth_escape_julia(pt, req.julia_c, req.max_iter),
                    |job| job.smooth_escape(pt, req.julia_c, req.max_iter),
                ),
            };
            values[flatten_2d_index(x, y, width)] = pot;
        }
    }
    ScalarField2D {
        width,
        height,
        values,
        periodic: false,
        default_levels: 10,
    }
}

impl FractalViewer {
    pub(super) fn sanitized_export_request(
        &self,
        width: usize,
        height: usize,
    ) -> FractalRenderRequest {
        let zoom = if self.zoom.is_finite() && (ZOOM_MIN..=ZOOM_MAX).contains(&self.zoom) {
            self.zoom
        } else {
            1.0
        };
        FractalRenderRequest {
            generation: self.request_generation,
            width,
            height,
            center: Complex::new(
                self.center.re.clamp(COORD_MIN, COORD_MAX),
                self.center.im.clamp(COORD_MIN, COORD_MAX),
            ),
            zoom,
            max_iter: self
                .max_iter
                .clamp(10, math_commons::registry::MAX_ITERATIONS as u32),
            mode: self.mode,
            julia_c: Complex::new(
                self.julia_c.re.clamp(COORD_MIN, COORD_MAX),
                self.julia_c.im.clamp(COORD_MIN, COORD_MAX),
            ),
            high_contrast: self.high_contrast,
        }
    }

    pub(super) fn compiled_export_formula(&self) -> Option<CompiledFractalJob> {
        self.mode
            .is_custom()
            .then(|| CompiledFractalJob::from_bar(&self.math_input))
    }
}
