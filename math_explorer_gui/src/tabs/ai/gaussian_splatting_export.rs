//! Offscreen figure, typed table, and INRIA-standard 3DGS binary `.ply` exporters for [`super::GaussianSplattingTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::scene::{rasterize_splats_into, SPLAT_GRID_SIZE, VIEWPORT_HALF_EXTENT};
use super::GaussianSplattingTool;
use crate::app::SemanticPalette;
use crate::framework::export_table::{ExportCell, ExportTable};
use eframe::egui;
use egui_plot::export_primitives::ExportPrimitive;
use nalgebra::Point3;
use oxidize_core::mesh::{export_splats_to_ply_binary, GaussianSplatPly};

fn safe_half_extent(half: f64) -> f64 {
    if half.is_finite() && half > 0.1 {
        half
    } else {
        VIEWPORT_HALF_EXTENT
    }
}

fn map_viewport_to_grid(pt: [f64; 2], x_half: f64, y_half: f64) -> [f64; 2] {
    let sx = safe_half_extent(x_half);
    let sy = safe_half_extent(y_half);
    let w = SPLAT_GRID_SIZE as f64;
    let gx = ((pt[0] + sx) / (2.0 * sx)) * w;
    let gy = ((pt[1] + sy) / (2.0 * sy)) * w;
    [gx, gy]
}

fn composite_over_backdrop(fg: egui::Color32, bg: egui::Color32) -> egui::Color32 {
    let inv_a = 1.0 - (f32::from(fg.a()) / 255.0);
    let r = (f32::from(fg.r()) + f32::from(bg.r()) * inv_a)
        .round()
        .clamp(0.0, 255.0) as u8;
    let g = (f32::from(fg.g()) + f32::from(bg.g()) * inv_a)
        .round()
        .clamp(0.0, 255.0) as u8;
    let b = (f32::from(fg.b()) + f32::from(bg.b()) * inv_a)
        .round()
        .clamp(0.0, 255.0) as u8;
    egui::Color32::from_rgb(r, g, b)
}

impl GaussianSplattingTool {
    pub(super) fn step_export_simulation(&mut self) {
        self.camera.yaw += 0.15;
        self.cached_key = None;
    }

    pub(super) fn build_export_raster_image(&self) -> egui::ColorImage {
        let mut projected = Vec::with_capacity(self.scene.gaussians.len());
        let mut rings = Vec::new();
        let mut rings_3d = Vec::new();
        let mut centers = Vec::new();
        self.project_into(&mut projected, &mut rings, &mut rings_3d, &mut centers);

        let mut pixels = vec![egui::Color32::TRANSPARENT; SPLAT_GRID_SIZE * SPLAT_GRID_SIZE];
        let (x_half, y_half) = self.frustum_half_extents;
        rasterize_splats_into(&projected, x_half, y_half, &mut pixels);

        let bg = egui::Visuals::dark().extreme_bg_color;
        for px in &mut pixels {
            *px = composite_over_backdrop(*px, bg);
        }
        egui::ColorImage {
            size: [SPLAT_GRID_SIZE, SPLAT_GRID_SIZE],
            source_size: egui::vec2(SPLAT_GRID_SIZE as f32, SPLAT_GRID_SIZE as f32),
            pixels,
        }
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = SemanticPalette::from_visuals(&egui::Visuals::dark());
        let mut projected = Vec::with_capacity(self.scene.gaussians.len());
        let mut rings = Vec::with_capacity(self.scene.gaussians.len());
        let mut rings_3d = Vec::new();
        let mut centers = Vec::with_capacity(self.scene.gaussians.len());
        self.project_into(&mut projected, &mut rings, &mut rings_3d, &mut centers);

        let (x_half, y_half) = self.frustum_half_extents;
        let mut prims = Vec::with_capacity(rings.len() + 5);

        let origin = map_viewport_to_grid(self.project_splat_point(Point3::origin()), x_half, y_half);
        let axes = [
            (Point3::new(1.2, 0.0, 0.0), palette.axis_x, "X"),
            (Point3::new(0.0, 1.2, 0.0), palette.axis_y, "Y"),
            (Point3::new(0.0, 0.0, 1.2), palette.axis_z, "Z"),
        ];
        for (tip_world, color, label) in axes {
            let tip = map_viewport_to_grid(self.project_splat_point(tip_world), x_half, y_half);
            prims.push(ExportPrimitive::Arrow {
                name: format!("Axis {label}"),
                origin,
                tip,
                tip_length: Some(8.0),
                color,
            });
        }

        for (idx, (ring, color)) in rings.iter().enumerate() {
            let points: Vec<[f64; 2]> = ring
                .iter()
                .map(|&pt| map_viewport_to_grid(pt, x_half, y_half))
                .collect();
            prims.push(ExportPrimitive::Polyline {
                name: format!("Splat {idx} Covariance"),
                points,
                color: *color,
                stroke_width: 1.5,
                closed: false,
                fill: None,
            });
        }

        let mapped_centers: Vec<[f64; 2]> = centers
            .iter()
            .map(|&pt| map_viewport_to_grid(pt, x_half, y_half))
            .collect();
        if !mapped_centers.is_empty() {
            prims.push(ExportPrimitive::Points {
                name: "Splat Means".into(),
                points: mapped_centers,
                color: egui::Color32::WHITE,
                radius: 3.0,
            });
        }
        prims
    }

    pub(super) fn build_export_tables(&self) -> Vec<ExportTable> {
        vec![self.build_3d_splat_table(), self.build_2d_splat_table()]
    }

    fn build_3d_splat_table(&self) -> ExportTable {
        let scale_factor = if self.scale.is_finite() {
            self.scale.clamp(0.1, 10.0)
        } else {
            1.0
        };
        let rows_3d: Vec<Vec<ExportCell>> = self
            .scene
            .gaussians
            .iter()
            .enumerate()
            .map(|(idx, g)| {
                let s = g.scale * scale_factor;
                vec![
                    ExportCell::UInt(idx as u64),
                    ExportCell::Float(g.mean.x),
                    ExportCell::Float(g.mean.y),
                    ExportCell::Float(g.mean.z),
                    ExportCell::Float(s.x),
                    ExportCell::Float(s.y),
                    ExportCell::Float(s.z),
                    ExportCell::Float(g.opacity),
                    ExportCell::Float(g.color.x),
                    ExportCell::Float(g.color.y),
                    ExportCell::Float(g.color.z),
                ]
            })
            .collect();
        ExportTable::new(
            "Gaussians 3D",
            [
                "splat_id", "mean_x", "mean_y", "mean_z", "scale_x", "scale_y", "scale_z",
                "opacity", "r", "g", "b",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            rows_3d,
        )
    }

    fn build_2d_splat_table(&self) -> ExportTable {
        let mut projected = Vec::with_capacity(self.scene.gaussians.len());
        let mut rings = Vec::new();
        let mut rings_3d = Vec::new();
        let mut centers = Vec::new();
        self.project_into(&mut projected, &mut rings, &mut rings_3d, &mut centers);

        let rows_2d: Vec<Vec<ExportCell>> = projected
            .iter()
            .enumerate()
            .map(|(rank, g)| {
                vec![
                    ExportCell::UInt(rank as u64),
                    ExportCell::Float(g.mean.x),
                    ExportCell::Float(g.mean.y),
                    ExportCell::Float(g.depth),
                    ExportCell::Float(g.color.x),
                    ExportCell::Float(g.color.y),
                    ExportCell::Float(g.color.z),
                    ExportCell::Float(g.opacity),
                ]
            })
            .collect();
        ExportTable::new(
            "Projected Splats 2D",
            [
                "depth_rank",
                "screen_x",
                "screen_y",
                "depth",
                "r",
                "g",
                "b",
                "opacity",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            rows_2d,
        )
    }

    /// Converts the active 3D Gaussian scene at `self.scale` into an INRIA-standard binary `.ply`.
    pub fn serialize_splats_ply_binary(&self) -> Result<Vec<u8>, String> {
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err("Gaussian splat scale must be finite and positive.".to_string());
        }
        let scale_factor = self.scale.clamp(0.1, 10.0);
        let mut splats = Vec::with_capacity(self.scene.gaussians.len());
        for g in &self.scene.gaussians {
            let record: GaussianSplatPly = g.to_splat_ply(scale_factor);
            splats.push(record);
        }
        export_splats_to_ply_binary(&splats).map_err(|e| e.to_string())
    }
}
