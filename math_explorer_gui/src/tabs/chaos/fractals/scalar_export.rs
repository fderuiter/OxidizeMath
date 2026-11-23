//! Target-resolution raster and smooth continuous escape-time 2D scalar-field exports for [`super::FractalViewer`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;

use super::formula::CompiledFractalJob;
use super::runner::{
    map_iter_to_color32, FractalMode, FractalRenderRequest, COORD_MAX, COORD_MIN, ZOOM_MAX,
    ZOOM_MIN,
};
use super::FractalViewer;
use crate::framework::{ScalarField2D, MAX_SYNTHESIZED_PARAMS};

const MAX_EXPORT_SIDE: usize = egui_plot::MAX_EXPORT_DIMENSION as usize;
const SMOOTH_BAILOUT_SQ: f64 = 256.0;

fn clamp_export_dims(width: usize, height: usize) -> (usize, usize) {
    if width == 0 || height == 0 {
        return (480, 320);
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

fn smooth_quadratic_escape(mut z: Complex<f64>, c: Complex<f64>, max_iter: u32) -> (u32, f64) {
    let mut iter = 0u32;
    while iter < max_iter {
        let r2 = z.norm_sqr();
        if !r2.is_finite() || r2 > SMOOTH_BAILOUT_SQ {
            let log_zn = 0.5 * r2.max(4.000_001).ln();
            let nu = (log_zn / std::f64::consts::LN_2).max(1e-12).ln() / std::f64::consts::LN_2;
            let smooth = (f64::from(iter) + 1.0 - nu).clamp(0.0, f64::from(max_iter));
            return (iter, smooth / f64::from(max_iter.max(1)));
        }
        z = z * z + c;
        iter += 1;
    }
    (max_iter, 1.0)
}

impl CompiledFractalJob {
    pub(super) fn smooth_escape(
        &self,
        z0: Complex<f64>,
        c: Complex<f64>,
        max_iter: u32,
    ) -> (u32, f64) {
        let mut vars = [Complex::new(0.0, 0.0); 2 + MAX_SYNTHESIZED_PARAMS];
        vars[0] = z0;
        vars[1] = c;
        for i in 0..self.param_len {
            vars[2 + i] = Complex::new(self.params[i], 0.0);
        }
        let total = 2 + self.param_len;
        let mut iter = 0u32;
        while iter < max_iter {
            let r2 = vars[0].norm_sqr();
            if !r2.is_finite() || r2 > SMOOTH_BAILOUT_SQ {
                let log_zn = 0.5 * r2.max(4.000_001).ln();
                let nu = (log_zn / std::f64::consts::LN_2).max(1e-12).ln() / std::f64::consts::LN_2;
                let smooth = (f64::from(iter) + 1.0 - nu).clamp(0.0, f64::from(max_iter));
                return (iter, smooth / f64::from(max_iter.max(1)));
            }
            vars[0] = self.compiled.eval_complex_slice(&vars[..total]);
            iter += 1;
        }
        (max_iter, 1.0)
    }
}

fn eval_smooth_point(
    pt: Complex<f64>,
    req: &FractalRenderRequest,
    formula: Option<&CompiledFractalJob>,
) -> (u32, f64) {
    match req.mode {
        FractalMode::Mandelbrot => {
            smooth_quadratic_escape(Complex::new(0.0, 0.0), pt, req.max_iter)
        }
        FractalMode::Julia => smooth_quadratic_escape(pt, req.julia_c, req.max_iter),
        FractalMode::CustomMandelbrot => formula.map_or_else(
            || smooth_quadratic_escape(Complex::new(0.0, 0.0), pt, req.max_iter),
            |job| job.smooth_escape(Complex::new(0.0, 0.0), pt, req.max_iter),
        ),
        FractalMode::CustomJulia => formula.map_or_else(
            || smooth_quadratic_escape(pt, req.julia_c, req.max_iter),
            |job| job.smooth_escape(pt, req.julia_c, req.max_iter),
        ),
    }
}

impl FractalViewer {
    fn build_export_request(&self, dims: (usize, usize)) -> FractalRenderRequest {
        let (width, height) = clamp_export_dims(dims.0, dims.1);
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

    pub(super) fn render_export_image_at(&self, target_dims: (usize, usize)) -> egui::ColorImage {
        let req = self.build_export_request(target_dims);
        let job = self
            .mode
            .is_custom()
            .then(|| CompiledFractalJob::from_bar(&self.math_input));
        let scale_y = 3.0 / req.zoom;
        let scale_x = scale_y * (req.width as f64 / req.height as f64);
        let x_start = req.center.re - scale_x * 0.5;
        let y_start = req.center.im + scale_y * 0.5;
        let mut pixels = vec![egui::Color32::BLACK; req.width * req.height];
        for y in 0..req.height {
            let im = y_start - (y as f64 / req.height as f64) * scale_y;
            for x in 0..req.width {
                let re = x_start + (x as f64 / req.width as f64) * scale_x;
                let (iter, _) = eval_smooth_point(Complex::new(re, im), &req, job.as_ref());
                pixels[flatten_2d_index(x, y, req.width)] =
                    map_iter_to_color32(iter, req.max_iter, req.high_contrast);
            }
        }
        egui::ColorImage::new([req.width, req.height], pixels)
    }

    pub(super) fn build_scalar_field_2d(
        &self,
        target_dims: Option<(usize, usize)>,
    ) -> ScalarField2D {
        let dims = target_dims.unwrap_or((480, 320));
        let req = self.build_export_request(dims);
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
                let (_, smooth) = eval_smooth_point(Complex::new(re, im), &req, job.as_ref());
                values[flatten_2d_index(x, y, req.width)] = smooth;
            }
        }
        ScalarField2D::continuous(req.width, req.height, values, false)
    }
}
