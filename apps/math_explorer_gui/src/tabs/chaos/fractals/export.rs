#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;

use crate::framework::{ExportCell, ExportTable};

use super::formula::CompiledFractalJob;
use super::runner::{
    clamp_fractal_dimensions, eval_escape_iterations, render_fractal_color_image_styled,
    snapshot_to_clamped_color_image, DEFAULT_EXPORT_VIEWPORT_SIZE, MAX_FRACTAL_TEXTURE_SIDE,
    ZOOM_MAX, ZOOM_MIN,
};
use super::FractalViewer;

impl FractalViewer {
    pub(super) fn active_export_dimensions(&self) -> (usize, usize) {
        let raw = if self.viewport_size.0 > 0 && self.viewport_size.1 > 0 {
            self.viewport_size
        } else {
            DEFAULT_EXPORT_VIEWPORT_SIZE
        };
        clamp_fractal_dimensions(raw.0, raw.1, MAX_FRACTAL_TEXTURE_SIDE)
    }

    pub(super) fn build_fractal_raster_export(&self) -> Option<egui::ColorImage> {
        let (width, height) = self.active_export_dimensions();
        if !self.dirty && self.completed_generation == self.request_generation {
            if let Some(snap) = self.worker.latest_snapshot() {
                if (snap.width, snap.height) == (width, height) {
                    if let Some(img) =
                        snapshot_to_clamped_color_image(snap, MAX_FRACTAL_TEXTURE_SIDE)
                    {
                        return Some(img);
                    }
                }
            }
        }
        let req = self.build_clamped_render_request(width, height);
        let job = self
            .mode
            .is_custom()
            .then(|| CompiledFractalJob::from_bar(&self.math_input));
        Some(render_fractal_color_image_styled(
            &req,
            job.as_ref(),
            self.color_style(),
        ))
    }

    pub(super) fn build_fractal_export_tables(&self) -> Vec<ExportTable> {
        let (raw_w, raw_h) = self.active_export_dimensions();
        let (width, height) = (raw_w.max(1), raw_h.max(1));
        let req = self.build_clamped_render_request(width, height);
        let job = self
            .mode
            .is_custom()
            .then(|| CompiledFractalJob::from_bar(&self.math_input));
        let zoom = if req.zoom.is_finite() && req.zoom >= ZOOM_MIN {
            req.zoom.min(ZOOM_MAX)
        } else {
            1.0
        };
        let scale_y = 3.0 / zoom;
        let scale_x = scale_y * (width as f64 / height as f64);
        let x_start = req.center.re - scale_x * 0.5;
        let y_start = req.center.im + scale_y * 0.5;
        let max_iter_f = f64::from(req.max_iter);
        let smooth = self.smooth_coloring;

        let table = ExportTable::from_bounded_grid_2d(
            "Escape Time Grid",
            width,
            height,
            vec![
                "x".to_string(),
                "y".to_string(),
                "re".to_string(),
                "im".to_string(),
                "escape_iterations".to_string(),
                "escaped".to_string(),
            ],
            |x, y| {
                let re = x_start + (x as f64 / width as f64) * scale_x;
                let im = y_start - (y as f64 / height as f64) * scale_y;
                let iter_val = eval_escape_iterations(re, im, &req, job.as_ref(), smooth);
                let escaped = iter_val.is_finite() && iter_val < max_iter_f;
                vec![
                    ExportCell::Int(x as i64),
                    ExportCell::Int(y as i64),
                    ExportCell::Float(re),
                    ExportCell::Float(im),
                    ExportCell::Float(iter_val),
                    ExportCell::Bool(escaped),
                ]
            },
        );
        vec![table]
    }
}
