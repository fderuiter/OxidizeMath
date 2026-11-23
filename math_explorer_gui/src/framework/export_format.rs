//! Internal format, target, and status types for deferred frame exports.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::sync::Arc;

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
pub(crate) type CustomExportWriter = Arc<dyn Fn(&str, &[u8]) + Send + Sync>;

/// Type alias for custom clipboard text writer function (useful for deterministic testing).
pub(crate) type CustomClipboardWriter = Arc<dyn Fn(&str) -> Result<(), String> + Send + Sync>;
