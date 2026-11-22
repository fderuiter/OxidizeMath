#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui::{Pos2, Rect, Vec2};

/// Computes the largest aspect-locked [`Rect`] centered inside `avail_rect` for a `grid_w x grid_h` raster.
#[must_use]
pub fn compute_aspect_locked_rect(avail_rect: Rect, grid_w: usize, grid_h: usize) -> Rect {
    if grid_w == 0 || grid_h == 0 || avail_rect.width() <= 0.0 || avail_rect.height() <= 0.0 {
        return Rect::from_min_size(avail_rect.min, Vec2::ZERO);
    }
    let grid_aspect = grid_w as f32 / grid_h as f32;
    let avail_w = avail_rect.width();
    let avail_h = avail_rect.height();
    let (target_w, target_h) = if avail_w / avail_h > grid_aspect {
        (avail_h * grid_aspect, avail_h)
    } else {
        (avail_w, avail_w / grid_aspect)
    };
    Rect::from_center_size(avail_rect.center(), Vec2::new(target_w, target_h))
}

/// Maps a screen pointer coordinate inside `rect` to discrete grid coordinates `(x, y)` using `floor()`.
///
/// Guarantees zero `+0.5`-cell rounding offset across the entire cell interior. When `flip_y` is
/// `true`, screen row `0` (top) maps to `grid_h - 1` to match bottom-up solver coordinates.
#[must_use]
pub fn map_pointer_to_cell(
    rect: Rect,
    pos: Pos2,
    grid_w: usize,
    grid_h: usize,
    flip_y: bool,
) -> Option<(usize, usize)> {
    if grid_w == 0
        || grid_h == 0
        || rect.width() <= 0.0
        || rect.height() <= 0.0
        || pos.x < rect.min.x
        || pos.x > rect.max.x
        || pos.y < rect.min.y
        || pos.y > rect.max.y
    {
        return None;
    }
    let u = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0 - f32::EPSILON);
    let v = ((pos.y - rect.min.y) / rect.height()).clamp(0.0, 1.0 - f32::EPSILON);
    let gx = ((u * grid_w as f32).floor() as usize).min(grid_w - 1);
    let img_row = ((v * grid_h as f32).floor() as usize).min(grid_h - 1);
    let gy = if flip_y {
        grid_h - 1 - img_row
    } else {
        img_row
    };
    Some((gx, gy))
}

/// Computes the screen-space [`Rect`] of discrete cell `(x, y)` within `canvas_rect`.
#[must_use]
pub fn cell_screen_rect(
    canvas_rect: Rect,
    x: usize,
    y: usize,
    grid_w: usize,
    grid_h: usize,
    flip_y: bool,
) -> Rect {
    if grid_w == 0 || grid_h == 0 {
        return Rect::from_min_size(canvas_rect.min, Vec2::ZERO);
    }
    let cx = x.min(grid_w - 1);
    let cy = y.min(grid_h - 1);
    let img_row = if flip_y { grid_h - 1 - cy } else { cy };
    let cell_w = canvas_rect.width() / grid_w as f32;
    let cell_h = canvas_rect.height() / grid_h as f32;
    let min = Pos2::new(
        canvas_rect.min.x + cx as f32 * cell_w,
        canvas_rect.min.y + img_row as f32 * cell_h,
    );
    Rect::from_min_size(min, Vec2::new(cell_w, cell_h))
}

/// Interpolates a gapless integer Bresenham line segment from `(x0, y0)` to `(x1, y1)`.
///
/// Bounded by `|x1 - x0| + |y1 - y0| + 1` iterations (NASA Power of 10 Rule 2).
#[must_use]
pub fn bresenham_line(x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<(i32, i32)> {
    let dx = (x1 - x0).abs();
    let dy_abs = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let dy = -dy_abs;
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;

    let max_steps = (dx as usize)
        .saturating_add(dy_abs as usize)
        .saturating_add(1);
    let mut points = Vec::with_capacity(max_steps);

    for _ in 0..max_steps {
        points.push((x, y));
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
    points
}
