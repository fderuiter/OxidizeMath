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

/// Declares dataset, figure, and analytical GLSL shader export capabilities for an [`super::InteractiveTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExportCapabilities {
    /// Whether tabular dataset exports (`CSV`, `JSON`) are supported.
    pub dataset: bool,
    /// Whether offscreen figure exports (`PNG`, `SVG`, `GIF`) are supported.
    pub figure: bool,
    /// Whether borderless 16-bit heightmap and normal-map PNG exports are supported.
    pub scalar_field: bool,
    /// Whether Marching Squares contour-level controls (`1..=16`) apply to SVG figure exports.
    pub contour_controls: bool,
    /// Whether standalone GLSL fragment shader exports (`WebGL2`, `Shadertoy`) are supported.
    pub shader: bool,
    /// Explanatory hover text shown when `CSV` / `JSON` buttons are disabled.
    pub dataset_disabled_reason: &'static str,
    /// Explanatory hover text shown when `PNG` / `SVG` / `GIF` buttons are disabled.
    pub figure_disabled_reason: &'static str,
    /// Explanatory reason surfaced when GLSL shader export is unavailable.
    pub shader_disabled_reason: &'static str,
}

impl Default for ExportCapabilities {
    fn default() -> Self {
        Self::plot()
    }
}

impl ExportCapabilities {
    /// Standard capability profile for tools backed by `egui_plot::Plot` series.
    #[must_use]
    pub const fn plot() -> Self {
        Self {
            dataset: true,
            figure: true,
            scalar_field: false,
            contour_controls: false,
            shader: false,
            dataset_disabled_reason: "",
            figure_disabled_reason: "",
            shader_disabled_reason: DEFAULT_SHADER_DISABLED_REASON,
        }
    }

    /// Capability profile for 2D continuous scalar-field simulations supporting Marching Squares SVG contours and 16-bit textures.
    #[must_use]
    pub const fn scalar_field() -> Self {
        Self {
            dataset: false,
            figure: true,
            scalar_field: true,
            contour_controls: true,
            shader: false,
            dataset_disabled_reason: "CSV and JSON exports are unavailable: this tool simulates a 2D spatial scalar field rather than tabular (X, Y) series.",
            figure_disabled_reason: "",
            shader_disabled_reason: DEFAULT_SHADER_DISABLED_REASON,
        }
    }

    /// Capability profile for discrete 2D lattice tools supporting exact vector `<path>` SVGs and borderless 16-bit textures.
    #[must_use]
    pub const fn discrete_lattice() -> Self {
        Self {
            dataset: false,
            figure: true,
            scalar_field: true,
            contour_controls: false,
            shader: false,
            dataset_disabled_reason: "CSV and JSON exports are unavailable: this tool renders a 2D discrete lattice rather than tabular (X, Y) series.",
            figure_disabled_reason: "",
            shader_disabled_reason: DEFAULT_SHADER_DISABLED_REASON,
        }
    }

    /// Capability profile for 2D matrix heatmap tools emitting exact `<rect>` SVG geometry and borderless 16-bit textures.
    #[must_use]
    pub const fn vector_grid() -> Self {
        Self {
            dataset: false,
            figure: true,
            scalar_field: true,
            contour_controls: false,
            shader: false,
            dataset_disabled_reason: "CSV and JSON exports are unavailable: this tool renders a 2D matrix heatmap rather than tabular (X, Y) series.",
            figure_disabled_reason: "",
            shader_disabled_reason: DEFAULT_SHADER_DISABLED_REASON,
        }
    }

    /// Capability profile for 2D raster/texture tools without tabular `(X, Y)` datasets.
    #[must_use]
    pub const fn raster_only() -> Self {
        Self {
            dataset: false,
            figure: true,
            scalar_field: false,
            contour_controls: false,
            shader: false,
            dataset_disabled_reason: "CSV and JSON exports are unavailable: this tool renders a 2D raster field rather than tabular (X, Y) series.",
            figure_disabled_reason: "",
            shader_disabled_reason: DEFAULT_SHADER_DISABLED_REASON,
        }
    }

    /// Capability profile for interactive calculators, text inspectors, or custom widgets without exportable series or figures.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            dataset: false,
            figure: false,
            scalar_field: false,
            contour_controls: false,
            shader: false,
            dataset_disabled_reason: "CSV and JSON exports are unavailable: this tool does not produce tabular plot series.",
            figure_disabled_reason: "Figure export is unavailable: this tool does not render an offscreen plot or raster field.",
            shader_disabled_reason: DEFAULT_SHADER_DISABLED_REASON,
        }
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
