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

/// Supported formats for deferred end-of-frame dataset and figure exports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ExportFormat {
    /// Comma-separated values dataset format (`.csv`).
    Csv,
    /// JavaScript Object Notation dataset format (`.json`).
    Json,
    /// High-resolution Portable Network Graphics figure format (`.png`).
    Png,
    /// Scalable Vector Graphics figure format (`.svg`).
    Svg,
    /// Animated Graphics Interchange Format (`.gif`).
    Gif,
    /// Standalone WebGL2 (`#version 300 es`) fragment shader (`.frag`).
    GlslWebGl2,
    /// Standalone Shadertoy (`mainImage`) fragment shader (`.frag`).
    GlslShadertoy,
}

impl ExportFormat {
    const fn meta(self) -> (&'static str, &'static str, &'static str, &'static str) {
        match self {
            Self::Csv => ("csv", "CSV", "CSV File", "dataset"),
            Self::Json => ("json", "JSON", "JSON File", "dataset"),
            Self::Png => ("png", "PNG", "PNG Image", "figure"),
            Self::Svg => ("svg", "SVG", "SVG Vector Image", "figure"),
            Self::Gif => ("gif", "GIF", "GIF Animation", "animation"),
            Self::GlslWebGl2 => (
                "frag",
                "WebGL2 GLSL",
                "WebGL2 GLSL Fragment Shader",
                "webgl2",
            ),
            Self::GlslShadertoy => (
                "frag",
                "Shadertoy GLSL",
                "Shadertoy Fragment Shader",
                "shadertoy",
            ),
        }
    }

    pub(super) const fn clipboard_item_noun(self) -> &'static str {
        match self {
            Self::GlslWebGl2 | Self::GlslShadertoy => "shader",
            _ => "dataset",
        }
    }

    /// Returns the standard file extension for this format.
    pub(crate) const fn extension(&self) -> &'static str {
        self.meta().0
    }

    /// Returns the uppercase label for display messages.
    pub(crate) const fn label(&self) -> &'static str {
        self.meta().1
    }

    /// Returns the file dialog filter description.
    pub(crate) const fn filter_name(&self) -> &'static str {
        self.meta().2
    }

    /// Returns the filename stem suffix (`dataset`, `figure`, or `animation`).
    pub(crate) const fn stem_suffix(&self) -> &'static str {
        self.meta().3
    }
}

/// Output destination target for deferred simulation dataset and figure export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ExportTarget {
    /// Save serialized dataset or figure to disk via file save dialog or browser download.
    File,
    /// Copy serialized dataset text or rendered figure image directly to the system clipboard.
    Clipboard,
}

/// A deferred export request combining format and output destination target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ExportRequest {
    /// Dataset or figure serialization format.
    pub(crate) format: ExportFormat,
    /// Output destination target.
    pub(crate) target: ExportTarget,
}

/// Inline status feedback for action-bar export operations.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ActionBarExportStatus {
    /// Human-readable confirmation or error message.
    pub(crate) message: String,
    /// Whether this status indicates an error or warning condition.
    pub(crate) is_error: bool,
    /// Optional `egui` timestamp (`ctx.input(|i| i.time)`) when temporary confirmation expires.
    pub(crate) expires_at_time: Option<f64>,
}

/// Type alias for custom export file writer function.
pub(crate) type CustomExportWriter = std::sync::Arc<dyn Fn(&str, &[u8]) + Send + Sync>;

/// Type alias for custom clipboard text writer function (useful for deterministic testing).
pub(crate) type CustomClipboardWriter =
    std::sync::Arc<dyn Fn(&str) -> Result<(), String> + Send + Sync>;
