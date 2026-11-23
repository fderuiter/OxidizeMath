#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:graph_parameters_rust]

use eframe::egui::{Color32, ColorImage};
use egui_plot::ExportPrimitive;
use math_explorer::math_kernel::colormap::{
    ensure_dark_canvas_contrast, heatmap_color, heatmap_color_high_contrast, mapped_gradient,
};
use math_explorer::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::analysis::pde::wave_2d::Wave2DSolver;

use super::{ChladniWaveTankTool, GRID_SIZE, WaveColormap};
use crate::app::palette::DARK_PALETTE;

impl ChladniWaveTankTool {
    pub(super) fn build_export_color_image(&self) -> ColorImage {
        let mut pixels = vec![Color32::BLACK; GRID_SIZE * GRID_SIZE];
        populate_field_pixels(&self.solver, self.state.colormap, &mut pixels);
        ColorImage {
            size: [GRID_SIZE, GRID_SIZE],
            pixels,
            ..Default::default()
        }
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let mut prims = Vec::new();
        if self.state.show_nodal_contours {
            for (i, &[p0, p1]) in self.contours.segments().iter().take(1_500).enumerate() {
                if p0[0].is_finite() && p0[1].is_finite() && p1[0].is_finite() && p1[1].is_finite()
                {
                    let name = if i == 0 { "Nodal Contours" } else { "" }.into();
                    let seg = [[p0[0] + 0.5, p0[1] + 0.5], [p1[0] + 0.5, p1[1] + 0.5]];
                    prims.push(ExportPrimitive::Polyline {
                        name,
                        points: egui_plot::SharedPoints::from_slice(&seg),
                        color: DARK_PALETTE.node_stroke,
                        stroke_width: 1.2,
                        closed: false,
                        fill: None,
                    });
                }
            }
        }
        if self.state.show_sand_particles {
            let pts: Vec<[f64; 2]> = self
                .sand
                .positions_x()
                .iter()
                .zip(self.sand.positions_y().iter())
                .step_by(2)
                .filter(|&(&px, &py)| px.is_finite() && py.is_finite())
                .map(|(&px, &py)| [px + 0.5, py + 0.5])
                .collect();
            if !pts.is_empty() {
                prims.push(ExportPrimitive::Points {
                    name: "Chladni Sand".into(),
                    points: pts.into(),
                    color: DARK_PALETTE.accent,
                    radius: 1.4,
                });
            }
        }
        if prims.is_empty() {
            let det_x = (self.state.detector_x as f64) + 0.5;
            prims.push(ExportPrimitive::Polyline {
                name: "Detector Screen".into(),
                points: egui_plot::SharedPoints::from_slice(&[
                    [det_x, 2.0],
                    [det_x, (GRID_SIZE - 2) as f64],
                ]),
                color: DARK_PALETTE.success,
                stroke_width: 1.5,
                closed: false,
                fill: None,
            });
        }
        prims
    }

    pub(super) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let finite_pairs = |vals: &[f64]| -> Vec<[f64; 2]> {
            vals.iter()
                .enumerate()
                .map(|(i, &v)| [i as f64, if v.is_finite() { v } else { 0.0 }])
                .collect()
        };
        let mid_y = GRID_SIZE / 2;
        let disp = self.solver.displacement();
        let wave_slice: Vec<[f64; 2]> = (0..GRID_SIZE)
            .map(|x| {
                let v = disp[flatten_2d_index(x, mid_y, GRID_SIZE)];
                [x as f64, if v.is_finite() { v } else { 0.0 }]
            })
            .collect();
        vec![
            (
                "detector_intensity".into(),
                finite_pairs(&self.detector_buf),
            ),
            (
                "radial_fft_spectrum".into(),
                finite_pairs(&self.radial_spectrum),
            ),
            ("centerline_wave_profile".into(), wave_slice),
        ]
    }
}

pub(super) fn populate_field_pixels(
    solver: &Wave2DSolver,
    colormap: WaveColormap,
    out_pixels: &mut [Color32],
) {
    let w = solver.width();
    let h = solver.height();
    let disp = solver.displacement();
    let energy = solver.energy_field();

    let mut max_abs_u = 1.0e-6_f64;
    let mut max_e = 1.0e-9_f64;
    for i in 0..disp.len() {
        let au = disp[i].abs();
        if au.is_finite() && au > max_abs_u {
            max_abs_u = au;
        }
        if energy[i].is_finite() && energy[i] > max_e {
            max_e = energy[i];
        }
    }

    for grid_y in 0..h {
        let img_row = (h - 1) - grid_y;
        for grid_x in 0..w {
            let src_idx = flatten_2d_index(grid_x, grid_y, w);
            let dst_idx = flatten_2d_index(grid_x, img_row, w);
            if solver.wall_mask[src_idx] <= 0.5 {
                out_pixels[dst_idx] = DARK_PALETTE.surface_fill;
                continue;
            }
            let rgb = match colormap {
                WaveColormap::PerceptualThermal => {
                    let t = (energy[src_idx] / max_e).sqrt().clamp(0.0, 1.0);
                    heatmap_color(t)
                }
                WaveColormap::BipolarPhase => mapped_gradient(disp[src_idx], -max_abs_u, max_abs_u),
                WaveColormap::WcagHighContrast => {
                    let t = (0.5 + 0.5 * (disp[src_idx] / max_abs_u)).clamp(0.0, 1.0);
                    ensure_dark_canvas_contrast(heatmap_color_high_contrast(t))
                }
            };
            let (dr, dg, db) = match solver.wave_speed_mask[src_idx] {
                s if s > 1.02 => (44, 30, 8),
                s if s < 0.98 => (16, 36, 56),
                _ => (0, 0, 0),
            };
            out_pixels[dst_idx] = Color32::from_rgb(
                rgb[0].saturating_add(dr),
                rgb[1].saturating_add(dg),
                rgb[2].saturating_add(db),
            );
        }
    }
}
