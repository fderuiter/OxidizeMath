//! Export capability declarations and status severity styling for [`super::InteractiveTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;

/// Target GLSL fragment shader dialect for analytical GPU exports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderDialect {
    /// Standalone WebGL2 / OpenGL ES 3.00 fragment shader (`#version 300 es`).
    WebGl2Es300,
    /// Standalone Shadertoy fragment shader (`void mainImage(out vec4 fragColor, in vec2 fragCoord)`).
    Shadertoy,
}

impl ShaderDialect {
    /// Human-readable label for UI actions and status confirmations.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::WebGl2Es300 => "WebGL2 GLSL",
            Self::Shadertoy => "Shadertoy GLSL",
        }
    }
}

const DEFAULT_SHADER_DISABLED_REASON: &str =
    "GLSL shader export is unavailable: this tool does not expose a closed-form analytical 2D fragment field.";
const DEFAULT_TOPOLOGY_DISABLED_REASON: &str =
    "Topology export is unavailable: this tool does not expose a discrete graph, network, or state-transition diagram.";

/// Declares dataset, figure, 3D mesh/splat, analytical GLSL shader, and graph topology export capabilities for an [`super::InteractiveTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExportCapabilities {
    /// Whether tabular dataset exports (`CSV`, `JSON`) are supported.
    pub dataset: bool,
    /// Whether offscreen figure exports (`PNG`, `SVG`, `GIF`) are supported.
    pub figure: bool,
    /// Whether watertight binary `.stl` 3D solid mesh export is supported.
    pub stl: bool,
    /// Whether binary `.ply` 3D mesh or Gaussian splat export is supported.
    pub ply: bool,
    /// Whether Wavefront `.obj` 3D mesh export is supported.
    pub obj: bool,
    /// Whether standalone GLSL fragment shader exports (`WebGL2`, `Shadertoy`) are supported.
    pub shader: bool,
    /// Whether discrete graph topology exports (`.dot`, `.graphml`, `TikZ`, `.svg`, `.csv`) are supported.
    pub topology: bool,
    /// Explanatory hover text shown when `CSV` / `JSON` buttons are disabled.
    pub dataset_disabled_reason: &'static str,
    /// Explanatory hover text shown when `PNG` / `SVG` / `GIF` buttons are disabled.
    pub figure_disabled_reason: &'static str,
    /// Explanatory hover text shown when `Export STL` is disabled.
    pub stl_disabled_reason: &'static str,
    /// Explanatory hover text shown when `Export PLY` is disabled.
    pub ply_disabled_reason: &'static str,
    /// Explanatory hover text shown when `Export OBJ` is disabled.
    pub obj_disabled_reason: &'static str,
    /// Explanatory reason surfaced when GLSL shader export is unavailable.
    pub shader_disabled_reason: &'static str,
    /// Explanatory reason surfaced when graph topology export is unavailable.
    pub topology_disabled_reason: &'static str,
}

impl Default for ExportCapabilities {
    fn default() -> Self {
        Self::plot()
    }
}

const DEFAULT_STL_DISABLED: &str =
    "STL export is unavailable: this tool does not generate a 3D triangle mesh.";
const DEFAULT_PLY_DISABLED: &str =
    "PLY export is unavailable: this tool does not generate a 3D mesh or Gaussian splat scene.";
const DEFAULT_OBJ_DISABLED: &str =
    "OBJ export is unavailable: this tool does not generate a 3D triangle mesh.";

impl ExportCapabilities {
    /// Standard capability profile for tools backed by `egui_plot::Plot` series.
    #[must_use]
    pub const fn plot() -> Self {
        Self {
            dataset: true,
            figure: true,
            stl: false,
            ply: false,
            obj: false,
            shader: false,
            topology: false,
            dataset_disabled_reason: "",
            figure_disabled_reason: "",
            stl_disabled_reason: DEFAULT_STL_DISABLED,
            ply_disabled_reason: DEFAULT_PLY_DISABLED,
            obj_disabled_reason: DEFAULT_OBJ_DISABLED,
            shader_disabled_reason: DEFAULT_SHADER_DISABLED_REASON,
            topology_disabled_reason: DEFAULT_TOPOLOGY_DISABLED_REASON,
        }
    }

    /// Capability profile for 2D raster/texture tools without tabular `(X, Y)` datasets.
    #[must_use]
    pub const fn raster_only() -> Self {
        Self {
            dataset: false,
            figure: true,
            stl: false,
            ply: false,
            obj: false,
            shader: false,
            topology: false,
            dataset_disabled_reason: "CSV and JSON exports are unavailable: this tool renders a 2D raster field rather than tabular (X, Y) series.",
            figure_disabled_reason: "",
            stl_disabled_reason: DEFAULT_STL_DISABLED,
            ply_disabled_reason: DEFAULT_PLY_DISABLED,
            obj_disabled_reason: DEFAULT_OBJ_DISABLED,
            shader_disabled_reason: DEFAULT_SHADER_DISABLED_REASON,
            topology_disabled_reason: DEFAULT_TOPOLOGY_DISABLED_REASON,
        }
    }

    /// Capability profile for analytical calculators that export tabular `CSV` and `JSON` datasets without offscreen plot or raster figures.
    #[cfg(test)]
    #[must_use]
    pub const fn dataset_only() -> Self {
        Self {
            dataset: true,
            figure: false,
            stl: false,
            ply: false,
            obj: false,
            shader: false,
            topology: false,
            dataset_disabled_reason: "",
            figure_disabled_reason: "Figure export is unavailable: this tool computes analytical tables rather than an offscreen plot or raster field.",
            stl_disabled_reason: DEFAULT_STL_DISABLED,
            ply_disabled_reason: DEFAULT_PLY_DISABLED,
            obj_disabled_reason: DEFAULT_OBJ_DISABLED,
            shader_disabled_reason: DEFAULT_SHADER_DISABLED_REASON,
            topology_disabled_reason: DEFAULT_TOPOLOGY_DISABLED_REASON,
        }
    }

    /// Capability profile for interactive calculators, text inspectors, or custom widgets without exportable series or figures.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            dataset: false,
            figure: false,
            stl: false,
            ply: false,
            obj: false,
            shader: false,
            topology: false,
            dataset_disabled_reason: "CSV and JSON exports are unavailable: this tool does not produce tabular plot series.",
            figure_disabled_reason: "Figure export is unavailable: this tool does not render an offscreen plot or raster field.",
            stl_disabled_reason: DEFAULT_STL_DISABLED,
            ply_disabled_reason: DEFAULT_PLY_DISABLED,
            obj_disabled_reason: DEFAULT_OBJ_DISABLED,
            shader_disabled_reason: DEFAULT_SHADER_DISABLED_REASON,
            topology_disabled_reason: DEFAULT_TOPOLOGY_DISABLED_REASON,
        }
    }

    /// Capability profile for network, graph, and circuit workbenches that export discrete topologies rather than 1D `(X, Y)` plot series.
    #[must_use]
    pub const fn topology_only() -> Self {
        let mut caps = Self::none();
        caps.topology = true;
        caps.topology_disabled_reason = "";
        caps.dataset_disabled_reason =
            "Plot CSV and JSON series are unavailable: use Topology CSV, DOT, or GraphML to export this network.";
        caps.figure_disabled_reason =
            "Plot PNG figure export is unavailable: use Topology SVG or LaTeX TikZ to export this network diagram.";
        caps
    }

    /// Enables `.dot`, `.graphml`, `TikZ`, `.svg`, and `.csv` topology exports alongside existing capabilities.
    #[must_use]
    pub const fn with_topology(mut self) -> Self {
        self.topology = true;
        self.topology_disabled_reason = "";
        self
    }

    /// Enables watertight `.stl`, `.ply`, and `.obj` 3D solid mesh exports alongside existing capabilities.
    #[must_use]
    pub const fn with_3d_solid(mut self) -> Self {
        self.stl = true;
        self.ply = true;
        self.obj = true;
        self.stl_disabled_reason = "";
        self.ply_disabled_reason = "";
        self.obj_disabled_reason = "";
        self
    }

    /// Capability profile for 3D Gaussian Splatting scenes supporting INRIA-standard `.ply` splat export.
    #[must_use]
    pub const fn splat_ply_only() -> Self {
        let mut caps = Self::none();
        caps.ply = true;
        caps.ply_disabled_reason = "";
        caps.stl_disabled_reason =
            "STL export is unavailable: Gaussian Splatting scenes represent volumetric splats rather than triangle meshes; use Export PLY.";
        caps.obj_disabled_reason =
            "OBJ export is unavailable: Gaussian Splatting scenes represent volumetric splats rather than triangle meshes; use Export PLY.";
        caps
    }

    /// Returns `true` when at least one 3D export format (`STL`, `PLY`, or `OBJ`) is supported.
    #[must_use]
    pub const fn has_3d_export(&self) -> bool {
        self.stl || self.ply || self.obj
    }

    /// Enables standalone WebGL2 and Shadertoy `.frag` shader exports on this capability profile.
    #[must_use]
    pub const fn with_shader(mut self) -> Self {
        self.shader = true;
        self.shader_disabled_reason = "";
        self
    }
}

/// Visual color and ARIA live-region urgency level for export outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExportStatusSeverity {
    /// Export succeeded with non-empty output.
    #[default]
    Success,
    /// Export was skipped or capped with a non-fatal warning.
    Warning,
    /// Export failed or rejected an empty payload.
    Error,
}

impl ExportStatusSeverity {
    /// Returns the default dark-theme WCAG 2.2 AA status label color from [`crate::app::palette::DARK_PALETTE`].
    #[must_use]
    pub const fn color(self) -> egui::Color32 {
        match self {
            Self::Success => crate::app::palette::DARK_PALETTE.success,
            Self::Warning => crate::app::palette::DARK_PALETTE.warning,
            Self::Error => crate::app::palette::DARK_PALETTE.error,
        }
    }

    /// Resolves the WCAG 2.2 AA theme-adaptive color from [`crate::app::SemanticPalette`].
    #[must_use]
    pub fn palette_color(self, visuals: &egui::Visuals) -> egui::Color32 {
        let palette = crate::app::SemanticPalette::from_visuals(visuals);
        match self {
            Self::Success => palette.success,
            Self::Warning => palette.warning,
            Self::Error => palette.error,
        }
    }

    /// Returns the ARIA live-region priority (`"polite"` for success, `"assertive"` for warnings/errors).
    #[must_use]
    pub const fn aria_priority(self) -> &'static str {
        match self {
            Self::Success => "polite",
            Self::Warning | Self::Error => "assertive",
        }
    }
}
