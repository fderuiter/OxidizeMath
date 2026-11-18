//! Low-level anti-aliased segment, dot, triangle, and alpha-compositing pixel operations.

pub(super) fn fill_triangle(
    pixels: &mut [egui::Color32],
    w: usize,
    h: usize,
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
    color: egui::Color32,
) {
    let min_x = (a.0.min(b.0).min(c.0).floor() as isize).clamp(0, w as isize - 1) as usize;
    let max_x = (a.0.max(b.0).max(c.0).ceil() as isize).clamp(0, w as isize - 1) as usize;
    let min_y = (a.1.min(b.1).min(c.1).floor() as isize).clamp(0, h as isize - 1) as usize;
    let max_y = (a.1.max(b.1).max(c.1).ceil() as isize).clamp(0, h as isize - 1) as usize;
    let edge = |p: (f64, f64), v0: (f64, f64), v1: (f64, f64)| -> f64 {
        (p.0 - v0.0) * (v1.1 - v0.1) - (p.1 - v0.1) * (v1.0 - v0.0)
    };
    let area = edge(c, a, b);
    if area.abs() < 1e-6 {
        return;
    }
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = (x as f64 + 0.5, y as f64 + 0.5);
            let (w0, w1, w2) = (edge(p, b, c), edge(p, c, a), edge(p, a, b));
            let inside = if area > 0.0 {
                w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0
            } else {
                w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0
            };
            if inside {
                pixels[y * w + x] = blend_over(pixels[y * w + x], color, 1.0);
            }
        }
    }
}

pub(super) fn draw_segment(
    pixels: &mut [egui::Color32],
    w: usize,
    h: usize,
    p0: (f64, f64),
    p1: (f64, f64),
    radius: f64,
    color: egui::Color32,
) {
    let min_x =
        ((p0.0.min(p1.0) - radius - 1.0).floor() as isize).clamp(0, w as isize - 1) as usize;
    let max_x = ((p0.0.max(p1.0) + radius + 1.0).ceil() as isize).clamp(0, w as isize - 1) as usize;
    let min_y =
        ((p0.1.min(p1.1) - radius - 1.0).floor() as isize).clamp(0, h as isize - 1) as usize;
    let max_y = ((p0.1.max(p1.1) + radius + 1.0).ceil() as isize).clamp(0, h as isize - 1) as usize;
    let (dx, dy) = (p1.0 - p0.0, p1.1 - p0.1);
    let len_sq = dx * dx + dy * dy;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
            let t = if len_sq == 0.0 {
                0.0
            } else {
                (((px - p0.0) * dx + (py - p0.1) * dy) / len_sq).clamp(0.0, 1.0)
            };
            let dist = ((px - (p0.0 + t * dx)).powi(2) + (py - (p0.1 + t * dy)).powi(2)).sqrt();
            let coverage = (radius + 0.5 - dist).clamp(0.0, 1.0);
            if coverage > 0.0 {
                pixels[y * w + x] = blend_over(pixels[y * w + x], color, coverage);
            }
        }
    }
}

pub(super) fn draw_dot(
    pixels: &mut [egui::Color32],
    w: usize,
    h: usize,
    p: (f64, f64),
    radius: f64,
    color: egui::Color32,
) {
    let min_x = ((p.0 - radius - 1.0).floor() as isize).clamp(0, w as isize - 1) as usize;
    let max_x = ((p.0 + radius + 1.0).ceil() as isize).clamp(0, w as isize - 1) as usize;
    let min_y = ((p.1 - radius - 1.0).floor() as isize).clamp(0, h as isize - 1) as usize;
    let max_y = ((p.1 + radius + 1.0).ceil() as isize).clamp(0, h as isize - 1) as usize;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let dist = ((x as f64 + 0.5 - p.0).powi(2) + (y as f64 + 0.5 - p.1).powi(2)).sqrt();
            let coverage = (radius + 0.5 - dist).clamp(0.0, 1.0);
            if coverage > 0.0 {
                pixels[y * w + x] = blend_over(pixels[y * w + x], color, coverage);
            }
        }
    }
}

pub(super) fn blend_over(
    dst: egui::Color32,
    src: egui::Color32,
    alpha_factor: f64,
) -> egui::Color32 {
    let src_a = (src.a() as f64 / 255.0) * alpha_factor.clamp(0.0, 1.0);
    if src_a <= 0.0 {
        return dst;
    }
    let dst_a = dst.a() as f64 / 255.0;
    let out_a = src_a + dst_a * (1.0 - src_a);
    if out_a <= 0.0 {
        return egui::Color32::TRANSPARENT;
    }
    let mix = |sc: u8, dc: u8| -> u8 {
        let out =
            ((sc as f64 / 255.0) * src_a + (dc as f64 / 255.0) * dst_a * (1.0 - src_a)) / out_a;
        (out * 255.0).round().clamp(0.0, 255.0) as u8
    };
    egui::Color32::from_rgba_unmultiplied(
        mix(src.r(), dst.r()),
        mix(src.g(), dst.g()),
        mix(src.b(), dst.b()),
        (out_a * 255.0).round().clamp(0.0, 255.0) as u8,
    )
}
