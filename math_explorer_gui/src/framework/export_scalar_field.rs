//! Cold-path 2D scalar-field contour SVG, 16-bit heightmap PNG, and PBR normal-map PNG export pipeline.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use egui_plot::export::{
    encode_heightmap_png_16bit, encode_normal_map_png, ExportOptions, MAX_EXPORT_DIMENSION,
};
use egui_plot::export_primitives::ExportPrimitive;
use math_commons::math_kernel::colormap::mapped_gradient;
use math_commons::math_kernel::types::flatten_2d_index;

use super::export_ui::export_action_button;
use super::*;

/// Geometric discretization topology of an exported 2D scalar field.
#[derive(Debug, Clone)]
pub enum ScalarFieldTopology {
    /// Continuous 2D field suitable for Marching Squares iso-contour extraction.
    Continuous {
        /// Optional active-domain mask (`false` marks solid walls or masked-out regions).
        active_mask: Option<Vec<bool>>,
    },
    /// Discrete integer/matrix cell lattice rendered as crisp vector SVG cells without interpolation.
    DiscreteLattice {
        /// Row-major `width * height` cell fill colors matching the tool's active palette.
        cell_colors: Vec<egui::Color32>,
    },
}

/// Cold-path 2D scalar field snapshot exposed by [`InteractiveTool::export_scalar_field_2d`].
#[derive(Debug, Clone)]
pub struct ScalarField2D {
    /// Grid width in cells/samples.
    pub width: usize,
    /// Grid height in cells/samples.
    pub height: usize,
    /// Row-major `width * height` scalar field samples (top row `y = 0` first).
    pub values: Vec<f64>,
    /// Whether opposite horizontal and vertical borders wrap toroidally.
    pub periodic: bool,
    /// Continuous contour vs. discrete lattice vectorization mode.
    pub topology: ScalarFieldTopology,
}

impl ScalarField2D {
    /// Constructs a continuous 2D scalar field snapshot.
    #[must_use]
    pub fn continuous(width: usize, height: usize, values: Vec<f64>, periodic: bool) -> Self {
        Self {
            width,
            height,
            values,
            periodic,
            topology: ScalarFieldTopology::Continuous { active_mask: None },
        }
    }

    /// Attaches an active-cell mask (`true` = active fluid/plate cell, `false` = obstacle/wall).
    #[must_use]
    pub fn with_active_mask(mut self, mask: Vec<bool>) -> Self {
        if let ScalarFieldTopology::Continuous { active_mask } = &mut self.topology {
            *active_mask = Some(mask);
        }
        self
    }

    /// Constructs a discrete lattice 2D field snapshot with per-cell vector colors.
    #[must_use]
    pub fn discrete_lattice(
        width: usize,
        height: usize,
        values: Vec<f64>,
        cell_colors: Vec<egui::Color32>,
    ) -> Self {
        Self {
            width,
            height,
            values,
            periodic: false,
            topology: ScalarFieldTopology::DiscreteLattice { cell_colors },
        }
    }
}

/// Resolves target `(width, height)` canvas pixel dimensions from [`ExportOptions`].
pub(crate) fn target_dimensions_for_options(
    options: &ExportOptions,
) -> Result<(usize, usize), String> {
    let scale = if options.scale.is_finite() && options.scale > 0.0 {
        options.scale
    } else {
        1.0
    };
    let target_w = ((options.width.max(1) as f32) * scale).round().max(1.0) as u32;
    let mut target_h = ((options.height.max(1) as f32) * scale).round().max(1.0) as u32;
    if let Some(ratio) = options
        .aspect_ratio
        .ratio()
        .filter(|&r| r.is_finite() && r > 0.0)
    {
        target_h = ((target_w as f32) / ratio).round().max(1.0) as u32;
    }
    if target_w > MAX_EXPORT_DIMENSION || target_h > MAX_EXPORT_DIMENSION {
        return Err(format!(
            "Requested dimensions ({target_w}x{target_h}) exceed maximum limit"
        ));
    }
    Ok((target_w as usize, target_h as usize))
}

fn sanitize_field_values(field: &ScalarField2D) -> Result<(Vec<f64>, f64, f64), String> {
    let expected = field
        .width
        .checked_mul(field.height)
        .ok_or_else(|| "Scalar field dimensions overflow".to_string())?;
    if field.width == 0 || field.height == 0 || field.values.len() != expected {
        return Err(format!(
            "Invalid scalar field dimensions {}x{} for {} samples",
            field.width,
            field.height,
            field.values.len()
        ));
    }
    let mask = match &field.topology {
        ScalarFieldTopology::Continuous { active_mask } => active_mask.as_deref(),
        ScalarFieldTopology::DiscreteLattice { .. } => None,
    };
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for (idx, &v) in field.values.iter().enumerate() {
        let active = mask.and_then(|m| m.get(idx).copied()).unwrap_or(true);
        if active && v.is_finite() {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    if !min_v.is_finite() || !max_v.is_finite() {
        min_v = 0.0;
        max_v = 0.0;
    }
    let out: Vec<f64> = field
        .values
        .iter()
        .enumerate()
        .map(|(idx, &v)| {
            let active = mask.and_then(|m| m.get(idx).copied()).unwrap_or(true);
            if active && v.is_finite() {
                v
            } else {
                min_v
            }
        })
        .collect();
    Ok((out, min_v, max_v))
}

#[cfg(feature = "pure_math")]
fn build_continuous_contour_primitives(
    tool_name: &str,
    field: &ScalarField2D,
    values: &[f64],
    min_v: f64,
    max_v: f64,
    options: &ExportOptions,
) -> Vec<ExportPrimitive> {
    use math_explorer::pure_math::analysis::pde::wave_2d::extract_stitched_contours;

    let w_f = field.width as f64;
    let h_f = field.height as f64;
    let mut prims = vec![ExportPrimitive::Bar {
        name: format!("{tool_name}_domain"),
        min: [0.0, 0.0],
        max: [w_f, h_f],
        fill: egui::Color32::TRANSPARENT,
        stroke_color: egui::Color32::TRANSPARENT,
        stroke_width: 0.0,
    }];
    let span = max_v - min_v;
    let levels = options.contour_levels.clamp(1, 64) as usize;
    let stroke_w = options.line_thickness.clamp(0.75, 6.0);
    if span > 1e-12 && field.width >= 2 && field.height >= 2 {
        let mask = match &field.topology {
            ScalarFieldTopology::Continuous { active_mask } => active_mask.as_deref(),
            ScalarFieldTopology::DiscreteLattice { .. } => None,
        };
        let iso_levels: Vec<f64> = (0..levels)
            .map(|step| min_v + ((step as f64 + 1.0) / (levels as f64 + 1.0)) * span)
            .collect();
        let sx = w_f / ((field.width - 1) as f64);
        let sy = h_f / ((field.height - 1) as f64);
        for (idx, chain) in
            extract_stitched_contours(values, mask, field.width, field.height, &iso_levels)
                .into_iter()
                .enumerate()
        {
            let frac = ((chain.iso_level - min_v) / span).clamp(0.0, 1.0);
            let [r, g, b] = mapped_gradient(frac, 0.0, 1.0);
            let points: Vec<[f64; 2]> = chain
                .points
                .into_iter()
                .map(|p| [p[0] * sx, h_f - p[1] * sy])
                .collect();
            prims.push(ExportPrimitive::ContourPath {
                name: format!("{tool_name}_iso_{idx}"),
                iso_level: chain.iso_level,
                points,
                closed: chain.closed,
                color: egui::Color32::from_rgb(r, g, b),
                stroke_width: stroke_w,
            });
        }
    }
    if prims.len() == 1 {
        let [r, g, b] = mapped_gradient(0.5, 0.0, 1.0);
        prims.push(ExportPrimitive::ContourPath {
            name: format!("{tool_name}_boundary"),
            iso_level: min_v,
            points: vec![[0.0, 0.0], [w_f, 0.0], [w_f, h_f], [0.0, h_f]],
            closed: true,
            color: egui::Color32::from_rgb(r, g, b),
            stroke_width: stroke_w,
        });
    }
    prims
}

#[cfg(not(feature = "pure_math"))]
fn build_continuous_contour_primitives(
    tool_name: &str,
    field: &ScalarField2D,
    _values: &[f64],
    min_v: f64,
    _max_v: f64,
    options: &ExportOptions,
) -> Vec<ExportPrimitive> {
    let w_f = field.width as f64;
    let h_f = field.height as f64;
    let [r, g, b] = mapped_gradient(0.5, 0.0, 1.0);
    vec![ExportPrimitive::ContourPath {
        name: format!("{tool_name}_boundary"),
        iso_level: min_v,
        points: vec![[0.0, 0.0], [w_f, 0.0], [w_f, h_f], [0.0, h_f]],
        closed: true,
        color: egui::Color32::from_rgb(r, g, b),
        stroke_width: options.line_thickness.clamp(0.75, 6.0),
    }]
}

fn build_discrete_lattice_primitives(
    tool_name: &str,
    width: usize,
    height: usize,
    cell_colors: &[egui::Color32],
) -> Result<Vec<ExportPrimitive>, String> {
    let expected = width
        .checked_mul(height)
        .ok_or_else(|| "Lattice dimensions overflow".to_string())?;
    if width == 0 || height == 0 || cell_colors.len() != expected {
        return Err(format!(
            "Invalid discrete lattice {}x{} with {} colors",
            width,
            height,
            cell_colors.len()
        ));
    }
    let base_fill = cell_colors[0];
    let skip_base = expected > 4096 && base_fill.a() > 0;
    let mut prims = Vec::new();
    if skip_base {
        prims.push(ExportPrimitive::Bar {
            name: format!("{tool_name}_lattice_bg"),
            min: [0.0, 0.0],
            max: [width as f64, height as f64],
            fill: base_fill,
            stroke_color: egui::Color32::TRANSPARENT,
            stroke_width: 0.0,
        });
    }
    let cell_label = format!("{tool_name}_lattice");
    for y in 0..height {
        let y_min = (height - 1 - y) as f64;
        let y_max = (height - y) as f64;
        let mut x = 0usize;
        while x < width {
            let c = cell_colors[flatten_2d_index(x, y, width)];
            let mut x_end = x + 1;
            while x_end < width && cell_colors[flatten_2d_index(x_end, y, width)] == c {
                x_end += 1;
            }
            if c.a() > 0 && !(skip_base && c == base_fill) {
                prims.push(ExportPrimitive::Bar {
                    name: cell_label.clone(),
                    min: [x as f64, y_min],
                    max: [x_end as f64, y_max],
                    fill: c,
                    stroke_color: egui::Color32::TRANSPARENT,
                    stroke_width: 0.0,
                });
            }
            x = x_end;
        }
    }
    Ok(prims)
}

/// Converts a [`ScalarField2D`] into vector SVG [`ExportPrimitive`]s (stitched `<path>` contours or crisp lattice cells).
pub(crate) fn scalar_field_to_svg_primitives(
    tool_name: &str,
    field: &ScalarField2D,
    options: &ExportOptions,
) -> Result<Vec<ExportPrimitive>, String> {
    let (values, min_v, max_v) = sanitize_field_values(field)?;
    match &field.topology {
        ScalarFieldTopology::Continuous { .. } => Ok(build_continuous_contour_primitives(
            tool_name, field, &values, min_v, max_v, options,
        )),
        ScalarFieldTopology::DiscreteLattice { cell_colors } => {
            build_discrete_lattice_primitives(tool_name, field.width, field.height, cell_colors)
        }
    }
}

/// Encodes a 16-bit grayscale heightmap PNG from an [`InteractiveTool`]'s 2D scalar field.
pub(crate) fn render_tool_heightmap_png(
    tool: &dyn InteractiveTool,
    options: &ExportOptions,
) -> Result<Vec<u8>, String> {
    let dims = target_dimensions_for_options(options)?;
    let target = if options.seamless_tile {
        None
    } else {
        Some(dims)
    };
    let field = tool
        .export_scalar_field_2d(target)
        .ok_or_else(|| format!("{} does not expose a 2D scalar field", tool.name()))?;
    let (values, _, _) = sanitize_field_values(&field)?;
    encode_heightmap_png_16bit(&values, field.width, field.height)
}

/// Encodes an 8-bit tangent-space PBR normal-map PNG from an [`InteractiveTool`]'s 2D scalar field.
pub(crate) fn render_tool_normal_map_png(
    tool: &dyn InteractiveTool,
    options: &ExportOptions,
) -> Result<Vec<u8>, String> {
    let dims = target_dimensions_for_options(options)?;
    let target = if options.seamless_tile {
        None
    } else {
        Some(dims)
    };
    let field = tool
        .export_scalar_field_2d(target)
        .ok_or_else(|| format!("{} does not expose a 2D scalar field", tool.name()))?;
    let (values, _, _) = sanitize_field_values(&field)?;
    encode_normal_map_png(
        &values,
        field.width,
        field.height,
        f64::from(options.normal_strength),
        field.periodic,
    )
}

impl SimulationFramework {
    /// Renders toolbar/menu action buttons for 2D scalar-field exports when supported by the active tool.
    pub(super) fn render_scalar_export_action_items(
        &mut self,
        ui: &mut egui::Ui,
        close_on_click: bool,
    ) {
        let supported = self
            .active_tool
            .as_ref()
            .is_some_and(|tool| tool.supports_scalar_field_2d());
        if !supported {
            return;
        }
        if export_action_button(
            ui,
            true,
            "Export SVG Contours",
            "Export vector SVG iso-contours or crisp lattice cells",
            close_on_click,
        ) {
            self.queue_export(ExportFormat::Svg);
        }
        if export_action_button(
            ui,
            true,
            "Export 16-bit Heightmap",
            "Export 16-bit grayscale displacement heightmap PNG",
            close_on_click,
        ) {
            self.queue_export(ExportFormat::HeightmapPng);
        }
        if export_action_button(
            ui,
            true,
            "Export Normal Map",
            "Export tangent-space PBR normal-map PNG",
            close_on_click,
        ) {
            self.queue_export(ExportFormat::NormalMapPng);
        }
    }

    /// Dispatches a queued `HeightmapPng` or `NormalMapPng` export request at end-of-frame.
    pub(super) fn dispatch_scalar_field_export(
        &mut self,
        format: ExportFormat,
        target: ExportTarget,
    ) {
        if target == ExportTarget::Clipboard {
            self.set_export_status(
                ExportStatusSeverity::Warning,
                "Clipboard export is unavailable for heightmap and normal-map textures; save PNG to disk.",
            );
            return;
        }
        let Some(tool_name) = self.ensure_figure_exportable() else {
            return;
        };
        let filename = Self::build_export_filename(&tool_name, format);
        let opts = self.export_options.clone();
        let tool = self
            .active_tool
            .as_ref()
            .expect("verified active tool")
            .as_ref();
        let res = match format {
            ExportFormat::HeightmapPng => render_tool_heightmap_png(tool, &opts),
            ExportFormat::NormalMapPng => render_tool_normal_map_png(tool, &opts),
            _ => Err("Unsupported scalar field export format".to_string()),
        };
        match res {
            Ok(bytes) => self.complete_file_export(
                format,
                &filename,
                &bytes,
                None,
                format.filter_name(),
                "Scalar field export failed",
            ),
            Err(err) => self.set_export_status(
                ExportStatusSeverity::Error,
                format!("Scalar field export failed: {err}"),
            ),
        }
    }
}
