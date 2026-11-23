//! Vector `<path>` SVG and 16-bit scalar-field export helpers for [`super::PrimeSpiralWidget`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use egui_plot::export_primitives::{export_contour_layers_svg, SvgContourLayer, SvgContourPath};
use math_commons::math_kernel::marching_squares::ScalarField2D;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::number_theory::primes::primes_up_to;

use super::PrimeSpiralWidget;

impl PrimeSpiralWidget {
    pub(super) fn build_ulam_scalar_field(size: usize) -> ScalarField2D {
        let num_cells = size * size;
        let mut values = vec![0.0_f64; num_cells];
        let mut lookup = vec![false; num_cells + 1];
        for p in primes_up_to(num_cells as u64) {
            if (p as usize) < lookup.len() {
                lookup[p as usize] = true;
            }
        }
        let half = (size / 2) as i32;
        let (mut x, mut y, mut dx, mut dy) = (0_i32, 0_i32, 0_i32, -1_i32);
        for (i, &prime_flag) in lookup.iter().enumerate().take(num_cells + 1).skip(1) {
            if (-half <= x) && (x <= half) && (-half <= y) && (y <= half) {
                let (px, py) = ((half + x) as usize, (half + y) as usize);
                if px < size && py < size {
                    if prime_flag {
                        values[flatten_2d_index(px, py, size)] = 1.0;
                    } else if i == 1 {
                        values[flatten_2d_index(px, py, size)] = 0.5;
                    }
                }
            }
            if x == y || (x < 0 && x == -y) || (x > 0 && x == 1 - y) {
                let temp = dx;
                dx = -dy;
                dy = temp;
            }
            x += dx;
            y += dy;
        }
        ScalarField2D {
            width: size,
            height: size,
            values,
            periodic: false,
            default_levels: 2,
        }
    }

    pub(super) fn build_ulam_vector_svg(
        size: usize,
        options: &egui_plot::export::ExportOptions,
    ) -> Result<String, String> {
        let field = Self::build_ulam_scalar_field(size);
        let bg_sum = u16::from(options.background_color.r())
            + u16::from(options.background_color.g())
            + u16::from(options.background_color.b());
        let palette = if bg_sum < 384 {
            crate::app::ThemeMode::Dark.palette()
        } else {
            crate::app::ThemeMode::Light.palette()
        };
        let mut prime_paths = Vec::new();
        let mut origin_paths = Vec::new();
        for py in 0..size {
            for px in 0..size {
                let v = field.values[flatten_2d_index(px, py, size)];
                if v >= 0.9 {
                    let (x0, y0) = (px as f64 + 0.12, py as f64 + 0.12);
                    let (x1, y1) = (px as f64 + 0.88, py as f64 + 0.88);
                    prime_paths.push(SvgContourPath {
                        points: vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                        closed: true,
                    });
                } else if v >= 0.4 {
                    let (cx, cy) = (px as f64 + 0.5, py as f64 + 0.5);
                    origin_paths.push(SvgContourPath {
                        points: vec![
                            [cx, cy - 0.45],
                            [cx + 0.45, cy],
                            [cx, cy + 0.45],
                            [cx - 0.45, cy],
                        ],
                        closed: true,
                    });
                }
            }
        }
        let layers = vec![
            SvgContourLayer {
                label: "ulam-primes".to_string(),
                color: palette.node_stroke,
                stroke_width: 0.75,
                fill: Some(palette.node_stroke),
                paths: prime_paths,
            },
            SvgContourLayer {
                label: "ulam-origin".to_string(),
                color: palette.axis_x,
                stroke_width: 1.0,
                fill: Some(palette.axis_x),
                paths: origin_paths,
            },
        ];
        export_contour_layers_svg(size as f64, size as f64, &layers, options)
    }
}
