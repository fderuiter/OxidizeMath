#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    COORD_MAX, COORD_MIN, FractalMode, FractalRenderRequest, FractalViewer, ZOOM_MAX, ZOOM_MIN,
    render_fractal_color_image,
};
use crate::app::SemanticPalette;
use eframe::egui;
use num_complex::Complex;

/// Exact raster dimensions (`160x120`) for the live Mandelbrot `J_c` Julia inset.
pub const JULIA_INSET_SIZE: (usize, usize) = (160, 120);
/// Number of quadratic iteration steps (`z_{n+1} = z_n^2 + c`) drawn from `z_0 = 0`.
pub const CRITICAL_ORBIT_STEPS: usize = 32;
/// Iteration cap for the synchronous `160x120` Julia preview so 60 FPS latency holds at 1080p.
pub const JULIA_INSET_MAX_ITER: u32 = 48;

const ORBIT_DRAW_NORM_SQR_CAP: f64 = 64.0;

/// Renders a `160x120` `J_c` preview image via `render_fractal_color_image`.
#[must_use]
pub fn render_julia_inset_image(
    julia_c: Complex<f64>,
    inset_iter: u32,
    high_contrast: bool,
) -> egui::ColorImage {
    let req = FractalRenderRequest {
        generation: 0,
        width: JULIA_INSET_SIZE.0,
        height: JULIA_INSET_SIZE.1,
        center: Complex::new(0.0, 0.0),
        zoom: 1.0,
        max_iter: inset_iter,
        mode: FractalMode::Julia,
        julia_c,
        high_contrast,
    };
    render_fractal_color_image(&req)
}

/// Maps a screen coordinate inside `rect` onto the complex plane `(re, im)`.
#[must_use]
pub fn screen_to_complex(
    pos: egui::Pos2,
    rect: egui::Rect,
    center: Complex<f64>,
    zoom: f64,
) -> Complex<f64> {
    let w = f64::from(rect.width()).max(1.0);
    let h = f64::from(rect.height()).max(1.0);
    let scale_y = 3.0 / zoom.clamp(ZOOM_MIN, ZOOM_MAX);
    let scale_x = scale_y * (w / h);
    let u = f64::from(pos.x - rect.left()) / w;
    let v = f64::from(pos.y - rect.top()) / h;
    Complex::new(
        (center.re + (u - 0.5) * scale_x).clamp(COORD_MIN, COORD_MAX),
        (center.im + (0.5 - v) * scale_y).clamp(COORD_MIN, COORD_MAX),
    )
}

/// Maps a complex coordinate `z` onto screen space inside `rect`.
#[must_use]
pub fn complex_to_screen(
    z: Complex<f64>,
    rect: egui::Rect,
    center: Complex<f64>,
    zoom: f64,
) -> egui::Pos2 {
    let w = f64::from(rect.width()).max(1.0);
    let h = f64::from(rect.height()).max(1.0);
    let scale_y = 3.0 / zoom.clamp(ZOOM_MIN, ZOOM_MAX);
    let scale_x = scale_y * (w / h);
    let u = (z.re - center.re) / scale_x + 0.5;
    let v = 0.5 - (z.im - center.im) / scale_y;
    egui::pos2(
        rect.left() + (u * w) as f32,
        rect.top() + (v * h) as f32,
    )
}

/// Computes up to `steps` iterates of the critical orbit `z_{n+1} = z_n^2 + c` from `z_0 = 0`.
#[must_use]
pub fn compute_critical_orbit(c: Complex<f64>, steps: usize) -> Vec<Complex<f64>> {
    let mut orbit = Vec::with_capacity(steps.saturating_add(1));
    let mut z = Complex::new(0.0, 0.0);
    orbit.push(z);
    if !c.re.is_finite() || !c.im.is_finite() {
        return orbit;
    }
    for _ in 0..steps {
        let next = z * z + c;
        if !next.re.is_finite() || !next.im.is_finite() || next.norm_sqr() > ORBIT_DRAW_NORM_SQR_CAP
        {
            break;
        }
        z = next;
        orbit.push(z);
    }
    orbit
}

impl FractalViewer {
    /// Refreshes the cached `160x120` `J_c` preview image via `render_fractal_color_image`.
    pub fn refresh_julia_inset(&mut self, ctx: &egui::Context) {
        let inset_iter = self.max_iter.clamp(10, JULIA_INSET_MAX_ITER);
        let safe_c = Complex::new(
            self.julia_c.re.clamp(COORD_MIN, COORD_MAX),
            self.julia_c.im.clamp(COORD_MIN, COORD_MAX),
        );
        let key = (
            safe_c.re.to_bits(),
            safe_c.im.to_bits(),
            inset_iter,
            self.high_contrast,
        );
        if self.julia_inset_key == Some(key) {
            if self.julia_inset_texture.is_none() {
                if let Some(img) = self.julia_inset_image.clone() {
                    self.julia_inset_texture = Some(ctx.load_texture(
                        "fractal_julia_inset",
                        img,
                        egui::TextureOptions::LINEAR,
                    ));
                }
            }
            return;
        }
        let img = render_julia_inset_image(safe_c, inset_iter, self.high_contrast);
        let opts = egui::TextureOptions::LINEAR;
        self.julia_inset_image = Some(img.clone());
        if let Some(tex) = &mut self.julia_inset_texture {
            tex.set(img, opts);
        } else {
            self.julia_inset_texture = Some(ctx.load_texture("fractal_julia_inset", img, opts));
        }
        self.julia_inset_key = Some(key);
    }

    pub(super) fn julia_inset_rect(canvas_rect: egui::Rect) -> Option<egui::Rect> {
        let margin = 10.0_f32;
        if canvas_rect.width() < 64.0 || canvas_rect.height() < 64.0 {
            return None;
        }
        let w = (JULIA_INSET_SIZE.0 as f32).min((canvas_rect.width() - margin * 2.0) * 0.48);
        let h = (JULIA_INSET_SIZE.1 as f32).min((canvas_rect.height() - margin * 2.0) * 0.48);
        if w < 24.0 || h < 18.0 {
            return None;
        }
        let min = egui::pos2(canvas_rect.right() - margin - w, canvas_rect.top() + margin);
        Some(egui::Rect::from_min_size(min, egui::vec2(w, h)))
    }

    pub(super) fn draw_mandelbrot_overlays(
        &mut self,
        ui: &egui::Ui,
        ctx: &egui::Context,
        canvas_rect: egui::Rect,
    ) {
        if self.mode != FractalMode::Mandelbrot {
            return;
        }
        self.refresh_julia_inset(ctx);
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let painter = ui.painter_at(canvas_rect);

        self.paint_critical_orbit(&painter, canvas_rect, &palette);
        self.paint_c_crosshair(&painter, canvas_rect, &palette);
        self.paint_julia_inset(&painter, canvas_rect, &palette);
    }

    fn paint_critical_orbit(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        palette: &SemanticPalette,
    ) {
        let orbit = compute_critical_orbit(self.julia_c, CRITICAL_ORBIT_STEPS);
        let screen_pts: Vec<egui::Pos2> = orbit
            .iter()
            .map(|&z| complex_to_screen(z, rect, self.center, self.zoom))
            .collect();
        let halo = egui::Stroke::new(2.8_f32, egui::Color32::from_black_alpha(180));
        let line_stroke = egui::Stroke::new(1.4_f32, palette.warning);
        for pair in screen_pts.windows(2) {
            painter.line_segment([pair[0], pair[1]], halo);
            painter.line_segment([pair[0], pair[1]], line_stroke);
        }
        for (idx, &pt) in screen_pts.iter().enumerate() {
            let r = if idx == 0 { 3.5_f32 } else { 2.3_f32 };
            painter.circle_filled(pt, r + 1.0_f32, egui::Color32::BLACK);
            painter.circle_filled(pt, r, palette.warning);
        }
    }

    fn paint_c_crosshair(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        palette: &SemanticPalette,
    ) {
        let c_pos = complex_to_screen(self.julia_c, rect, self.center, self.zoom);
        let arm = if self.c_pinned { 12.0_f32 } else { 9.0_f32 };
        let halo = egui::Stroke::new(3.2_f32, egui::Color32::from_black_alpha(200));
        let fg = egui::Stroke::new(
            if self.c_pinned { 2.0_f32 } else { 1.5_f32 },
            palette.accent,
        );

        let h_seg = [
            egui::pos2(c_pos.x - arm, c_pos.y),
            egui::pos2(c_pos.x + arm, c_pos.y),
        ];
        let v_seg = [
            egui::pos2(c_pos.x, c_pos.y - arm),
            egui::pos2(c_pos.x, c_pos.y + arm),
        ];
        painter.line_segment(h_seg, halo);
        painter.line_segment(v_seg, halo);
        painter.line_segment(h_seg, fg);
        painter.line_segment(v_seg, fg);
        if self.c_pinned {
            painter.circle_stroke(c_pos, 5.5, halo);
            painter.circle_stroke(c_pos, 5.5, fg);
        }
    }

    fn paint_julia_inset(
        &self,
        painter: &egui::Painter,
        canvas_rect: egui::Rect,
        palette: &SemanticPalette,
    ) {
        let (Some(inset_rect), Some(tex)) =
            (Self::julia_inset_rect(canvas_rect), &self.julia_inset_texture)
        else {
            return;
        };
        let frame_rect = inset_rect.expand(2.0);
        painter.rect_filled(frame_rect, 3.0, egui::Color32::from_black_alpha(210));
        let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        painter.image(tex.id(), inset_rect, uv, egui::Color32::WHITE);
        painter.rect_stroke(
            frame_rect,
            3.0_f32,
            egui::Stroke::new(
                if self.c_pinned { 2.0_f32 } else { 1.2_f32 },
                palette.accent,
            ),
            egui::StrokeKind::Inside,
        );
        let caption = format!(
            "J_c ({:+.2}{:+.2}i)",
            self.julia_c.re, self.julia_c.im
        );
        painter.text(
            egui::pos2(inset_rect.left() + 4.0, inset_rect.bottom() - 4.0),
            egui::Align2::LEFT_BOTTOM,
            caption,
            egui::FontId::proportional(11.0),
            egui::Color32::WHITE,
        );
    }
}
