#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::PrimeSpiralWidget;
use crate::framework::{compute_matrix_stride, ExportCell, ExportTable};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::number_theory::primes::primes_up_to;

fn compute_spiral_buffers(size: usize) -> (Vec<u32>, Vec<bool>) {
    let num_cells = size * size;
    let mut is_prime = vec![false; num_cells + 1];
    for p in primes_up_to(num_cells as u64) {
        if let Some(slot) = is_prime.get_mut(p as usize) {
            *slot = true;
        }
    }
    let mut grid = vec![0u32; num_cells];
    let half = (size / 2) as i32;
    let (mut x, mut y, mut dx, mut dy) = (0i32, 0i32, 0i32, -1i32);
    for i in 1..=num_cells {
        if (-half <= x) && (x <= half) && (-half <= y) && (y <= half) {
            let (px, py) = ((half + x) as usize, (half + y) as usize);
            if px < size && py < size {
                grid[flatten_2d_index(px, py, size)] = i as u32;
            }
        }
        if x == y || (x < 0 && x == -y) || (x > 0 && x == 1 - y) {
            let tmp = dx;
            dx = -dy;
            dy = tmp;
        }
        x += dx;
        y += dy;
    }
    (grid, is_prime)
}

pub(super) fn build_ulam_export_tables(widget: &PrimeSpiralWidget) -> Vec<ExportTable> {
    let size = widget.grid_size.max(2);
    let num_cells = size * size;
    let cached_ready =
        widget.spiral_grid.len() == num_cells && widget.is_prime_lookup.len() > num_cells;
    let owned;
    let (grid, lookup): (&[u32], &[bool]) = if cached_ready {
        (&widget.spiral_grid, &widget.is_prime_lookup)
    } else {
        owned = compute_spiral_buffers(size);
        (&owned.0, &owned.1)
    };

    let half = i64::from((size / 2) as i32);
    let (r_stride, c_stride) = compute_matrix_stride(size, size);
    let mut rows = Vec::new();
    for py in (0..size).step_by(r_stride) {
        for px in (0..size).step_by(c_stride) {
            let val = u64::from(grid[flatten_2d_index(px, py, size)]);
            let prime = lookup.get(val as usize).copied().unwrap_or(false);
            rows.push(vec![
                ExportCell::UInt(py as u64),
                ExportCell::UInt(px as u64),
                ExportCell::Int(px as i64 - half),
                ExportCell::Int(py as i64 - half),
                ExportCell::UInt(val),
                ExportCell::Bool(prime),
            ]);
        }
    }
    vec![ExportTable::new(
        "Ulam Spiral",
        vec![
            "row".into(),
            "col".into(),
            "x".into(),
            "y".into(),
            "value".into(),
            "is_prime".into(),
        ],
        rows,
    )]
}
