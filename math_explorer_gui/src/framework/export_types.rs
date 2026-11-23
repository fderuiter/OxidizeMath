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
            shader: false,
            dataset_disabled_reason: "",
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
            shader: false,
            dataset_disabled_reason: "CSV and JSON exports are unavailable: this tool does not produce tabular plot series.",
            figure_disabled_reason: "Figure export is unavailable: this tool does not render an offscreen plot or raster field.",
            shader_disabled_reason: DEFAULT_SHADER_DISABLED_REASON,
        }
    }

    /// Capability profile for tabular calculators and text inspectors without offscreen plots or figures.
    #[must_use]
    pub const fn dataset_only() -> Self {
        Self {
            dataset: true,
            figure: false,
            shader: false,
            dataset_disabled_reason: "",
            figure_disabled_reason: "Figure export is unavailable: this tool produces tabular results rather than an offscreen plot or visual scene.",
            shader_disabled_reason: DEFAULT_SHADER_DISABLED_REASON,
        }
    }

    /// Capability profile for tools awaiting user input or execution before producing exportable data or figures.
    #[must_use]
    pub const fn unpopulated(reason: &'static str) -> Self {
        Self {
            dataset: false,
            figure: false,
            shader: false,
            dataset_disabled_reason: reason,
            figure_disabled_reason: reason,
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
