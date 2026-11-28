//! Cold-path INRIA `.ply`, `.png`, `.svg`, and `.csv` exporters for [`super::GaussianSplattingTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:gaussian_splatting]

use std::fmt::Write as _;
use std::sync::Arc;

use eframe::egui;
use egui_plot::export::{color32_image_to_png, ExportOptions};
use egui_plot::export_primitives::{
    export_primitives_svg, render_primitives_image, ExportPrimitive,
};
use oxidize_core::mesh::{export_splats_to_ply_binary, GaussianSplatPly};

use super::scene::{
    build_circle_ring, rasterize_splats_into, SplatLensConfig, SplatLensMode,
    VolumetricSplatPreset, DEFAULT_GRAD_THRESHOLD, DEFAULT_OPACITY_THRESHOLD, GRAD_THRESHOLD_MAX,
    GRAD_THRESHOLD_MIN, OPACITY_THRESHOLD_MAX, OPACITY_THRESHOLD_MIN, SPLAT_GRID_SIZE,
};
use super::GaussianSplattingTool;
use crate::framework::{Camera3D, ExportCell, ExportTable, InteractiveTool};

const fn default_grad_threshold() -> f64 {
    DEFAULT_GRAD_THRESHOLD
}

const fn default_opacity_threshold() -> f64 {
    DEFAULT_OPACITY_THRESHOLD
}

#[derive(serde::Serialize)]
struct GaussianSplattingSaveState {
    scale: f64,
    grad_threshold: f64,
    opacity_threshold: f64,
    camera: Camera3D,
    #[serde(skip_serializing_if = "VolumetricSplatPreset::is_canonical")]
    scene_preset: VolumetricSplatPreset,
    #[serde(skip_serializing_if = "SplatLensConfig::is_default")]
    lens: SplatLensConfig,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct GaussianSplattingState {
    scale: f64,
    #[serde(default = "default_grad_threshold")]
    grad_threshold: f64,
    #[serde(default = "default_opacity_threshold")]
    opacity_threshold: f64,
    #[serde(default)]
    scene_preset: VolumetricSplatPreset,
    #[serde(default)]
    lens: SplatLensConfig,
    #[serde(default)]
    camera: Option<Camera3D>,
    #[serde(default)]
    pitch: Option<f32>,
    #[serde(default)]
    yaw: Option<f32>,
    #[serde(default)]
    zoom: Option<f32>,
}

impl GaussianSplattingTool {
    pub(super) fn serialize_tool_state_json(&self) -> Option<String> {
        serde_json::to_string(&GaussianSplattingSaveState {
            scale: self.scale,
            grad_threshold: self.grad_threshold,
            opacity_threshold: self.opacity_threshold,
            camera: self.camera,
            scene_preset: self.scene_preset,
            lens: self.lens,
        })
        .ok()
    }

    pub(super) fn deserialize_tool_state_json(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: GaussianSplattingState =
            serde_json::from_str(state).map_err(|e| format!("Invalid splatting state: {e}"))?;
        if !parsed.scale.is_finite() || !(0.1..=10.0).contains(&parsed.scale) {
            return Err("Gaussian splat scale must be finite within [0.1, 10.0]".into());
        }
        if !parsed.grad_threshold.is_finite()
            || !(GRAD_THRESHOLD_MIN..=GRAD_THRESHOLD_MAX).contains(&parsed.grad_threshold)
        {
            return Err("Gradient threshold out of valid bounds".into());
        }
        if !parsed.opacity_threshold.is_finite()
            || !(OPACITY_THRESHOLD_MIN..=OPACITY_THRESHOLD_MAX).contains(&parsed.opacity_threshold)
        {
            return Err("Opacity threshold out of valid bounds".into());
        }
        if !parsed.lens.is_valid() {
            return Err("Gravitational lens parameters out of valid bounds".into());
        }
        let camera = Camera3D::from_preset_fields(
            parsed.camera,
            parsed.pitch,
            parsed.yaw,
            parsed.zoom,
            Some(Camera3D::default()),
        )?;
        self.scale = parsed.scale;
        self.grad_threshold = parsed.grad_threshold;
        self.opacity_threshold = parsed.opacity_threshold;
        self.lens = parsed.lens;
        self.camera.apply_pose(camera);
        self.set_scene_preset(parsed.scene_preset);
        self.sync_density_counts();
        self.invalidate_render_cache();
        Ok(())
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

    /// Renders the current lensed or unlensed 2D radiance field as an offscreen [`egui::ColorImage`].
    pub fn render_current_splat_image(&self) -> egui::ColorImage {
        let dual_cap = self.scene.gaussians.len().saturating_mul(2).max(16);
        let mut projected = Vec::with_capacity(dual_cap);
        let mut rings = Vec::with_capacity(64);
        let mut rings_3d = Vec::with_capacity(64);
        let mut centers = Vec::with_capacity(dual_cap);
        let _ = self.project_into(&mut projected, &mut rings, &mut rings_3d, &mut centers);
        let mut pixels = vec![egui::Color32::TRANSPARENT; SPLAT_GRID_SIZE * SPLAT_GRID_SIZE];
        let (x_half, y_half) = self.frustum_half_extents;
        rasterize_splats_into(&projected, x_half, y_half, &mut pixels);
        egui::ColorImage {
            size: [SPLAT_GRID_SIZE, SPLAT_GRID_SIZE],
            source_size: egui::vec2(SPLAT_GRID_SIZE as f32, SPLAT_GRID_SIZE as f32),
            pixels,
        }
    }

    /// Encodes the current lensed radiance field and vector caustic overlays to PNG bytes.
    pub fn export_lensed_png_bytes(&self) -> Result<Vec<u8>, String> {
        let prims = self.build_lensed_figure_primitives();
        let opts = ExportOptions {
            background_color: egui::Visuals::dark().extreme_bg_color,
            ..Default::default()
        };
        let image = render_primitives_image(&prims, &opts)?;
        color32_image_to_png(&image)
    }

    /// Exports the rasterized lensed radiance field overlaid with vector covariance ellipses and
    /// Einstein / NFW / Burkert caustic rings as an SVG document.
    pub fn export_lensed_svg_string(&self) -> Result<String, String> {
        let prims = self.build_lensed_figure_primitives();
        let opts = ExportOptions {
            background_color: egui::Visuals::dark().extreme_bg_color,
            ..Default::default()
        };
        export_primitives_svg(&prims, &opts)
    }

    pub(super) fn build_lensed_figure_primitives(&self) -> Vec<ExportPrimitive> {
        let dual_cap = self.scene.gaussians.len().saturating_mul(2).max(16);
        let mut projected = Vec::with_capacity(dual_cap);
        let mut rings = Vec::with_capacity(64);
        let mut rings_3d = Vec::with_capacity(64);
        let mut centers = Vec::with_capacity(dual_cap);
        let telemetry = self.project_into(&mut projected, &mut rings, &mut rings_3d, &mut centers);
        let mut pixels = vec![egui::Color32::TRANSPARENT; SPLAT_GRID_SIZE * SPLAT_GRID_SIZE];
        let (x_half, y_half) = self.frustum_half_extents;
        rasterize_splats_into(&projected, x_half, y_half, &mut pixels);
        let raster = egui::ColorImage {
            size: [SPLAT_GRID_SIZE, SPLAT_GRID_SIZE],
            source_size: egui::vec2(SPLAT_GRID_SIZE as f32, SPLAT_GRID_SIZE as f32),
            pixels,
        };

        let palette = crate::app::SemanticPalette::from_visuals(&egui::Visuals::dark());
        let mut prims = Vec::new();
        prims.push(ExportPrimitive::RasterImage {
            name: "Lensed Radiance Field".to_string(),
            image: Arc::new(raster),
            bounds: Some([-x_half, x_half, -y_half, y_half]),
        });
        for (ring_pts, color) in rings.iter().take(32) {
            prims.push(ExportPrimitive::Polyline {
                name: "Projected Covariance".to_string(),
                points: ring_pts.to_vec().into(),
                color: *color,
                stroke_width: 1.2,
                closed: true,
                fill: None,
            });
        }
        self.append_svg_caustic_primitives(&telemetry, &palette, &mut prims);
        prims
    }

    fn append_svg_caustic_primitives(
        &self,
        telemetry: &super::scene::LensedFrameTelemetry,
        palette: &crate::app::SemanticPalette,
        prims: &mut Vec<ExportPrimitive>,
    ) {
        if self.lens.mode == SplatLensMode::Off || !self.lens.show_caustic_overlay {
            return;
        }
        let center = self.lens.center;
        if self.lens.mode == SplatLensMode::Schwarzschild {
            prims.push(ExportPrimitive::Polyline {
                name: "Einstein Ring".to_string(),
                points: build_circle_ring(center, self.lens.einstein_radius)
                    .to_vec()
                    .into(),
                color: palette.warning,
                stroke_width: 1.8,
                closed: true,
                fill: None,
            });
            prims.push(ExportPrimitive::Polyline {
                name: "Photon Sphere Shadow (1.5 R_s)".to_string(),
                points: build_circle_ring(center, telemetry.shadow_radius)
                    .to_vec()
                    .into(),
                color: palette.error,
                stroke_width: 1.6,
                closed: true,
                fill: None,
            });
        }
        if let Some(r_nfw) = telemetry.nfw_critical_radius {
            prims.push(ExportPrimitive::Polyline {
                name: "NFW Cusp Critical Curve".to_string(),
                points: build_circle_ring(center, r_nfw).to_vec().into(),
                color: palette.accent,
                stroke_width: 1.5,
                closed: true,
                fill: None,
            });
        }
        if let Some(r_burk) = telemetry.burkert_critical_radius {
            prims.push(ExportPrimitive::Polyline {
                name: "Burkert Core Critical Curve".to_string(),
                points: build_circle_ring(center, r_burk).to_vec().into(),
                color: palette.success,
                stroke_width: 1.5,
                closed: true,
                fill: None,
            });
        }
    }

    /// Exports projected/lensed 2D Gaussian splats, conic matrices, and caustic radii as CSV.
    pub fn export_lensed_csv_string(&self) -> Result<String, String> {
        let dual_cap = self.scene.gaussians.len().saturating_mul(2).max(16);
        let mut projected = Vec::with_capacity(dual_cap);
        let mut rings = Vec::new();
        let mut rings_3d = Vec::new();
        let mut centers = Vec::with_capacity(dual_cap);
        let telemetry = self.project_into(&mut projected, &mut rings, &mut rings_3d, &mut centers);

        let mut csv = String::with_capacity(projected.len().saturating_mul(96) + 256);
        csv.push_str(
            "index,lens_mode,mean_x,mean_y,depth,opacity,color_r,color_g,color_b,conic_xx,conic_xy,conic_yy,shadow_radius,nfw_crit_r,burkert_crit_r\n",
        );
        let mode_str = match self.lens.mode {
            SplatLensMode::Off => "off",
            SplatLensMode::Schwarzschild => "schwarzschild",
            SplatLensMode::NfwCusp => "nfw_cusp",
            SplatLensMode::BurkertCore => "burkert_core",
        };
        let nfw_r = telemetry.nfw_critical_radius.unwrap_or(0.0);
        let burk_r = telemetry.burkert_critical_radius.unwrap_or(0.0);
        for (idx, g) in projected.iter().enumerate() {
            let _ = writeln!(
                csv,
                "{idx},{mode_str},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                g.mean.x,
                g.mean.y,
                g.depth,
                g.opacity,
                g.color.x,
                g.color.y,
                g.color.z,
                g.conic[(0, 0)],
                g.conic[(0, 1)],
                g.conic[(1, 1)],
                telemetry.shadow_radius,
                nfw_r,
                burk_r,
            );
        }
        Ok(csv)
    }

    fn build_projected_splats_table(&self) -> ExportTable {
        let dual_cap = self.scene.gaussians.len().saturating_mul(2).max(16);
        let mut projected = Vec::with_capacity(dual_cap);
        let (mut rings, mut rings_3d) = (Vec::new(), Vec::new());
        let mut centers = Vec::with_capacity(dual_cap);
        let telemetry = self.project_into(&mut projected, &mut rings, &mut rings_3d, &mut centers);
        let mode_str = match self.lens.mode {
            SplatLensMode::Off => "off",
            SplatLensMode::Schwarzschild => "schwarzschild",
            SplatLensMode::NfwCusp => "nfw_cusp",
            SplatLensMode::BurkertCore => "burkert_core",
        };
        let (nfw_r, burk_r) = (
            telemetry.nfw_critical_radius.unwrap_or(0.0),
            telemetry.burkert_critical_radius.unwrap_or(0.0),
        );
        let splat_rows = projected
            .iter()
            .enumerate()
            .map(|(idx, g)| {
                vec![
                    ExportCell::Int(idx as i64),
                    ExportCell::Text(mode_str.to_string()),
                    ExportCell::Float(g.mean.x),
                    ExportCell::Float(g.mean.y),
                    ExportCell::Float(g.depth),
                    ExportCell::Float(g.opacity),
                    ExportCell::Float(g.color.x),
                    ExportCell::Float(g.color.y),
                    ExportCell::Float(g.color.z),
                    ExportCell::Float(g.conic[(0, 0)]),
                    ExportCell::Float(g.conic[(0, 1)]),
                    ExportCell::Float(g.conic[(1, 1)]),
                    ExportCell::Float(telemetry.shadow_radius),
                    ExportCell::Float(nfw_r),
                    ExportCell::Float(burk_r),
                ]
            })
            .collect();
        let headers = [
            "index",
            "lens_mode",
            "mean_x",
            "mean_y",
            "depth",
            "opacity",
            "color_r",
            "color_g",
            "color_b",
            "conic_xx",
            "conic_xy",
            "conic_yy",
            "shadow_radius",
            "nfw_crit_r",
            "burkert_crit_r",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        ExportTable::new("Projected Splats", headers, splat_rows)
    }

    pub(super) fn build_splat_export_tables(&self) -> Vec<ExportTable> {
        let counts = evaluate_density_summary(self);
        let summary_headers = [
            "scale",
            "grad_threshold",
            "opacity_threshold",
            "split_count",
            "clone_count",
            "keep_count",
            "prune_count",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let summary_table = ExportTable::new(
            "Adaptive Density Summary",
            summary_headers,
            vec![vec![
                ExportCell::Float(self.scale),
                ExportCell::Float(self.grad_threshold),
                ExportCell::Float(self.opacity_threshold),
                ExportCell::Int(counts.split as i64),
                ExportCell::Int(counts.clone as i64),
                ExportCell::Int(counts.keep as i64),
                ExportCell::Int(counts.prune as i64),
            ]],
        );
        vec![self.build_projected_splats_table(), summary_table]
    }
}

fn evaluate_density_summary(tool: &GaussianSplattingTool) -> super::DensityActionCounts {
    super::scene::evaluate_scene_density_counts_with_thresholds(
        &tool.scene,
        &tool.grad_magnitudes,
        tool.scale,
        tool.grad_threshold,
        tool.opacity_threshold,
    )
}
