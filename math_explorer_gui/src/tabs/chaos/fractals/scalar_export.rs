//! Target-resolution raster and smooth continuous escape-time 2D scalar-field exports for [`super::FractalViewer`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::chaos::fractals::{smooth_escape_julia, smooth_escape_mandelbrot};
use num_complex::Complex;

use super::formula::CompiledFractalJob;
use super::runner::{
    clamp_fractal_dimensions, eval_pixel_color, render_fractal_color_image_styled,
    snapshot_to_clamped_color_image, FractalMode, FractalRenderRequest,
    DEFAULT_EXPORT_VIEWPORT_SIZE, MAX_FRACTAL_TEXTURE_SIDE,
};
use super::FractalViewer;
use crate::framework::ScalarField2D;

const MAX_EXPORT_SIDE: usize = egui_plot::MAX_EXPORT_DIMENSION as usize;

fn clamp_export_dims(width: usize, height: usize) -> (usize, usize) {
    if width == 0 || height == 0 {
        return DEFAULT_EXPORT_VIEWPORT_SIZE;
    }
    let max_dim = width.max(height);
    if max_dim <= MAX_EXPORT_SIDE {
        return (width, height);
    }
    let scale = (MAX_EXPORT_SIDE as f64) / (max_dim as f64);
    let cw = ((width as f64 * scale).round() as usize).clamp(1, MAX_EXPORT_SIDE);
    let ch = ((height as f64 * scale).round() as usize).clamp(1, MAX_EXPORT_SIDE);
    (cw, ch)
}

fn eval_normalized_smooth_escape(
    pt: Complex<f64>,
    req: &FractalRenderRequest,
    formula: Option<&CompiledFractalJob>,
) -> f64 {
    let raw = match req.mode {
        FractalMode::Mandelbrot => smooth_escape_mandelbrot(pt, req.max_iter),
        FractalMode::Julia => smooth_escape_julia(pt, req.julia_c, req.max_iter),
        FractalMode::CustomMandelbrot => formula.map_or_else(
            || smooth_escape_mandelbrot(pt, req.max_iter),
            |job| job.smooth_escape_time(Complex::new(0.0, 0.0), pt, req.max_iter),
        ),
        FractalMode::CustomJulia => formula.map_or_else(
            || smooth_escape_julia(pt, req.julia_c, req.max_iter),
            |job| job.smooth_escape_time(pt, req.julia_c, req.max_iter),
        ),
    };
    (raw / f64::from(req.max_iter.max(1))).clamp(0.0, 1.0)
}

impl FractalViewer {
    fn native_viewport_dims(&self) -> (usize, usize) {
        let raw = if self.viewport_size.0 > 0 && self.viewport_size.1 > 0 {
            self.viewport_size
        } else {
            DEFAULT_EXPORT_VIEWPORT_SIZE
        };
        clamp_fractal_dimensions(raw.0, raw.1, MAX_FRACTAL_TEXTURE_SIDE)
    }

    pub(super) fn export_viewport_or_fallback_image(&self) -> egui::ColorImage {
        let (width, height) = self.native_viewport_dims();
        if !self.dirty && self.completed_generation == self.request_generation {
            if let Some(snap) = self.worker.latest_snapshot() {
                if (snap.width, snap.height) == (width, height) {
                    if let Some(img) =
                        snapshot_to_clamped_color_image(snap, MAX_FRACTAL_TEXTURE_SIDE)
                    {
                        return img;
                    }
                }
            }
        }
        let req = self.build_clamped_render_request(width, height);
        let job = self
            .mode
            .is_custom()
            .then(|| CompiledFractalJob::from_bar(&self.math_input));
        render_fractal_color_image_styled(&req, job.as_ref(), self.color_style())
    }

    pub(super) fn render_export_image_at(&self, target_dims: (usize, usize)) -> egui::ColorImage {
        let (width, height) = clamp_export_dims(target_dims.0, target_dims.1);
        let req = self.build_clamped_render_request(width, height);
        let job = self
            .mode
            .is_custom()
            .then(|| CompiledFractalJob::from_bar(&self.math_input));
        let style = self.color_style();
        let scale_y = 3.0 / req.zoom;
        let scale_x = scale_y * (req.width as f64 / req.height as f64);
        let x_start = req.center.re - scale_x * 0.5;
        let y_start = req.center.im + scale_y * 0.5;
        let mut pixels = vec![egui::Color32::BLACK; req.width * req.height];
        for y in 0..req.height {
            let im = y_start - (y as f64 / req.height as f64) * scale_y;
            for x in 0..req.width {
                let re = x_start + (x as f64 / req.width as f64) * scale_x;
                pixels[flatten_2d_index(x, y, req.width)] =
                    eval_pixel_color(re, im, &req, job.as_ref(), style);
            }
        }
        egui::ColorImage::new([req.width, req.height], pixels)
    }

    pub(super) fn build_scalar_field_2d(
        &self,
        target_dims: Option<(usize, usize)>,
    ) -> ScalarField2D {
        let (width, height) = target_dims.map_or_else(
            || self.native_viewport_dims(),
            |(w, h)| clamp_export_dims(w, h),
        );
        let req = self.build_clamped_render_request(width, height);
        let job = self
            .mode
            .is_custom()
            .then(|| CompiledFractalJob::from_bar(&self.math_input));
        let scale_y = 3.0 / req.zoom;
        let scale_x = scale_y * (req.width as f64 / req.height as f64);
        let x_start = req.center.re - scale_x * 0.5;
        let y_start = req.center.im + scale_y * 0.5;
        let mut values = vec![0.0; req.width * req.height];
        for y in 0..req.height {
            let im = y_start - (y as f64 / req.height as f64) * scale_y;
            for x in 0..req.width {
                let re = x_start + (x as f64 / req.width as f64) * scale_x;
                values[flatten_2d_index(x, y, req.width)] =
                    eval_normalized_smooth_escape(Complex::new(re, im), &req, job.as_ref());
            }
        }
        ScalarField2D::continuous(req.width, req.height, values, false)
    }
}
