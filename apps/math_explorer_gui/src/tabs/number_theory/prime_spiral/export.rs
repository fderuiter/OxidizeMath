#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{ExportCell, ExportTable};
use math_commons::math_kernel::types::flatten_2d_index;

use super::PrimeSpiralWidget;

pub(super) fn build_prime_spiral_tables(tool: &PrimeSpiralWidget) -> Vec<ExportTable> {
    let size = tool.grid_size.max(2);
    let num_pixels = size * size;
    let half = (size / 2) as i64;

    let mut fallback_lookup = Vec::new();
    let mut fallback_pixels = Vec::new();
    let mut fallback_coords = Vec::new();
    let mut fallback_grid = Vec::new();

    let (grid, lookup) =
        if tool.spiral_grid.len() == num_pixels && tool.is_prime_lookup.len() > num_pixels {
            (&tool.spiral_grid, &tool.is_prime_lookup)
        } else {
            PrimeSpiralWidget::populate_ulam_buffers(
                size,
                crate::app::ThemeMode::Dark,
                &mut fallback_lookup,
                &mut fallback_pixels,
                &mut fallback_coords,
                &mut fallback_grid,
            );
            (&fallback_grid, &fallback_lookup)
        };

    let table = ExportTable::from_bounded_grid_2d(
        "Ulam Prime Spiral",
        size,
        size,
        vec![
            "x".to_string(),
            "y".to_string(),
            "dx".to_string(),
            "dy".to_string(),
            "value".to_string(),
            "is_prime".to_string(),
        ],
        |x, y| {
            let val = grid.get(flatten_2d_index(x, y, size)).copied().unwrap_or(0);
            let prime = lookup.get(val as usize).copied().unwrap_or(false);
            vec![
                ExportCell::Int(x as i64),
                ExportCell::Int(y as i64),
                ExportCell::Int(x as i64 - half),
                ExportCell::Int(y as i64 - half),
                ExportCell::Int(i64::from(val)),
                ExportCell::Bool(prime),
            ]
        },
    );
    vec![table]
}
