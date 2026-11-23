#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    CompiledFractalJob, FractalMode, FractalViewer, COORD_MAX, COORD_MIN, ZOOM_MAX, ZOOM_MIN,
};
use crate::framework::{compute_matrix_stride, ExportCell, ExportTable};
use math_explorer::physics::chaos::fractals::{escape_time_julia, escape_time_mandelbrot};
use num_complex::Complex;

impl FractalViewer {
    pub(super) fn build_fractal_export_tables(&self) -> Vec<ExportTable> {
        let (req_w, req_h) = self.requested_size;
        let width = req_w.max(120);
        let height = req_h.max(90);
        let (r_stride, c_stride) = compute_matrix_stride(height, width);
        let zoom = if self.zoom.is_finite() && (ZOOM_MIN..=ZOOM_MAX).contains(&self.zoom) {
            self.zoom
        } else {
            1.0
        };
        let center_re = self.center.re.clamp(COORD_MIN, COORD_MAX);
        let center_im = self.center.im.clamp(COORD_MIN, COORD_MAX);
        let julia_c = Complex::new(
            self.julia_c.re.clamp(COORD_MIN, COORD_MAX),
            self.julia_c.im.clamp(COORD_MIN, COORD_MAX),
        );
        let max_iter = self
            .max_iter
            .clamp(10, math_commons::registry::MAX_ITERATIONS as u32);
        let scale_y = 3.0 / zoom;
        let aspect = (width as f64) / (height.max(1) as f64);
        let scale_x = scale_y * aspect;
        let custom_job = self
            .mode
            .is_custom()
            .then(|| CompiledFractalJob::from_bar(&self.math_input));

        let mut rows = Vec::new();
        for row in (0..height).step_by(r_stride) {
            let im = center_im + ((row as f64) / (height as f64) - 0.5) * scale_y;
            for col in (0..width).step_by(c_stride) {
                let re = center_re + ((col as f64) / (width as f64) - 0.5) * scale_x;
                let pt = Complex::new(re, im);
                let iters =
                    eval_point_iterations(self.mode, pt, julia_c, max_iter, custom_job.as_ref());
                rows.push(vec![
                    ExportCell::UInt(row as u64),
                    ExportCell::UInt(col as u64),
                    ExportCell::Float(re),
                    ExportCell::Float(im),
                    ExportCell::UInt(u64::from(iters)),
                    ExportCell::Bool(iters < max_iter),
                ]);
            }
        }
        vec![ExportTable::new(
            "Escape Iterations",
            vec![
                "row".into(),
                "col".into(),
                "re".into(),
                "im".into(),
                "iterations".into(),
                "escaped".into(),
            ],
            rows,
        )]
    }
}

fn eval_point_iterations(
    mode: FractalMode,
    pt: Complex<f64>,
    julia_c: Complex<f64>,
    max_iter: u32,
    job: Option<&CompiledFractalJob>,
) -> u32 {
    match mode {
        FractalMode::Mandelbrot => escape_time_mandelbrot(pt, max_iter),
        FractalMode::Julia => escape_time_julia(pt, julia_c, max_iter),
        FractalMode::CustomMandelbrot => {
            job.map_or(0, |j| j.escape_time(Complex::new(0.0, 0.0), pt, max_iter))
        }
        FractalMode::CustomJulia => job.map_or(0, |j| j.escape_time(pt, julia_c, max_iter)),
    }
}
