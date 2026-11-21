use super::theme::ThemeMode;
use eframe::egui::{self, Color32};

/// Stack-allocated semantic color palette resolved from active [`egui::Visuals`].
///
/// Every text, badge, axis, and categorical series token meets WCAG 2.2 SC 1.4.3
/// `>= 4.5:1` contrast against `panel_fill`, `window_fill`, and `extreme_bg_color`
/// across [`ThemeMode::Dark`], [`ThemeMode::Light`], and [`ThemeMode::HighContrast`].
/// Structural stroke tokens meet `>= 3.0:1` graphical contrast across all themes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticPalette {
    /// Informational status and active worker badge color (`>= 4.5:1`).
    pub info: Color32,
    /// Success, ready, and parity-verified status color (`>= 4.5:1`).
    pub success: Color32,
    /// Warning, transitional, and stepping status color (`>= 4.5:1`).
    pub warning: Color32,
    /// Error, faulted, and validation alert color (`>= 4.5:1`).
    pub error: Color32,
    /// Fatal and critical severity badge color (`>= 4.5:1`).
    pub fatal: Color32,
    /// Muted / secondary label and inactive status color (`>= 4.5:1`).
    pub muted_text: Color32,
    /// Neutral elevated surface / cell base / divider track fill.
    pub surface_fill: Color32,
    /// Numerical divergence / recovery banner background fill.
    pub error_banner_bg: Color32,
    /// Numerical divergence / recovery banner border stroke (`>= 3.0:1`).
    pub error_banner_stroke: Color32,
    /// Graph edge, wireframe, reference line, and hovered divider color (`>= 3.0:1`).
    pub graph_edge: Color32,
    /// High-contrast node / gizmo ring outline stroke (`>= 3.0:1`).
    pub node_stroke: Color32,
    /// X-axis / primary red vector and channel color (`>= 4.5:1`).
    pub axis_x: Color32,
    /// Y-axis / primary green vector and channel color (`>= 4.5:1`).
    pub axis_y: Color32,
    /// Z-axis / primary blue vector and channel color (`>= 4.5:1`).
    pub axis_z: Color32,
    /// Highlight accent for state vectors, active dividers, and callout leaders (`>= 4.5:1`).
    pub accent: Color32,
    /// Categorical plot series colors (`>= 4.5:1` for both curves and colored labels).
    pub series: [Color32; 6],
}

/// Semantic palette tuned for [`ThemeMode::Dark`].
pub const DARK_PALETTE: SemanticPalette = SemanticPalette {
    info: Color32::from_rgb(96, 182, 255),
    success: Color32::from_rgb(74, 214, 122),
    warning: Color32::from_rgb(245, 194, 66),
    error: Color32::from_rgb(255, 118, 118),
    fatal: Color32::from_rgb(255, 102, 153),
    muted_text: Color32::from_rgb(176, 182, 192),
    surface_fill: Color32::from_rgb(46, 50, 58),
    error_banner_bg: Color32::from_rgb(42, 18, 18),
    error_banner_stroke: Color32::from_rgb(235, 85, 85),
    graph_edge: Color32::from_rgb(132, 140, 152),
    node_stroke: Color32::from_rgb(230, 234, 240),
    axis_x: Color32::from_rgb(255, 108, 108),
    axis_y: Color32::from_rgb(76, 214, 112),
    axis_z: Color32::from_rgb(98, 172, 255),
    accent: Color32::from_rgb(250, 204, 56),
    series: [
        Color32::from_rgb(96, 182, 255),
        Color32::from_rgb(255, 138, 76),
        Color32::from_rgb(74, 214, 122),
        Color32::from_rgb(255, 112, 132),
        Color32::from_rgb(196, 148, 255),
        Color32::from_rgb(245, 198, 66),
    ],
};

/// Semantic palette tuned for [`ThemeMode::Light`] (classroom projection and print).
pub const LIGHT_PALETTE: SemanticPalette = SemanticPalette {
    info: Color32::from_rgb(0, 92, 178),
    success: Color32::from_rgb(15, 115, 54),
    warning: Color32::from_rgb(146, 84, 0),
    error: Color32::from_rgb(180, 28, 28),
    fatal: Color32::from_rgb(136, 19, 55),
    muted_text: Color32::from_rgb(82, 88, 98),
    surface_fill: Color32::from_rgb(224, 228, 235),
    error_banner_bg: Color32::from_rgb(254, 236, 236),
    error_banner_stroke: Color32::from_rgb(185, 28, 28),
    graph_edge: Color32::from_rgb(104, 112, 124),
    node_stroke: Color32::from_rgb(38, 44, 54),
    axis_x: Color32::from_rgb(190, 28, 28),
    axis_y: Color32::from_rgb(16, 118, 52),
    axis_z: Color32::from_rgb(12, 88, 184),
    accent: Color32::from_rgb(152, 82, 0),
    series: [
        Color32::from_rgb(12, 88, 184),
        Color32::from_rgb(176, 68, 0),
        Color32::from_rgb(15, 115, 54),
        Color32::from_rgb(180, 28, 56),
        Color32::from_rgb(112, 42, 172),
        Color32::from_rgb(138, 90, 0),
    ],
};

/// Semantic palette tuned for [`ThemeMode::HighContrast`].
pub const HIGH_CONTRAST_PALETTE: SemanticPalette = SemanticPalette {
    info: Color32::from_rgb(110, 205, 255),
    success: Color32::from_rgb(80, 250, 123),
    warning: Color32::from_rgb(255, 230, 0),
    error: Color32::from_rgb(255, 105, 105),
    fatal: Color32::from_rgb(255, 128, 191),
    muted_text: Color32::from_rgb(224, 224, 224),
    surface_fill: Color32::from_rgb(32, 32, 32),
    error_banner_bg: Color32::BLACK,
    error_banner_stroke: Color32::from_rgb(255, 105, 105),
    graph_edge: Color32::from_rgb(196, 196, 196),
    node_stroke: Color32::WHITE,
    axis_x: Color32::from_rgb(255, 105, 105),
    axis_y: Color32::from_rgb(80, 250, 123),
    axis_z: Color32::from_rgb(110, 205, 255),
    accent: Color32::from_rgb(255, 230, 0),
    series: [
        Color32::from_rgb(110, 205, 255),
        Color32::from_rgb(255, 168, 76),
        Color32::from_rgb(80, 250, 123),
        Color32::from_rgb(255, 105, 140),
        Color32::from_rgb(214, 158, 255),
        Color32::from_rgb(255, 230, 0),
    ],
};

impl SemanticPalette {
    /// Returns the constant semantic palette corresponding to `mode` with zero heap allocations.
    #[inline]
    pub const fn for_theme(mode: ThemeMode) -> Self {
        match mode {
            ThemeMode::Dark => DARK_PALETTE,
            ThemeMode::Light => LIGHT_PALETTE,
            ThemeMode::HighContrast => HIGH_CONTRAST_PALETTE,
        }
    }

    /// Resolves the active semantic palette directly from `visuals`, including split sub-panes.
    #[inline]
    pub fn from_visuals(visuals: &egui::Visuals) -> Self {
        Self::for_theme(ThemeMode::from_visuals(visuals))
    }

    /// Returns a categorical plot series color, wrapping cleanly across arbitrary series counts.
    #[inline]
    pub const fn series_color(self, index: usize) -> Color32 {
        self.series[index % self.series.len()]
    }

    /// Returns a severity fill for [`egui::ProgressBar`] that preserves `>= 4.5:1` contrast
    /// against `visuals.selection.stroke.color` and `>= 3.0:1` against dark tracks.
    #[inline]
    #[must_use]
    pub fn progress_bar_fill(visuals: &egui::Visuals, elasticity: f64) -> Color32 {
        if visuals.dark_mode {
            if elasticity > 2.0 {
                Color32::from_rgb(172, 46, 46)
            } else if elasticity > 1.0 {
                Color32::from_rgb(130, 82, 0)
            } else {
                Color32::from_rgb(24, 108, 54)
            }
        } else if elasticity > 2.0 {
            Color32::from_rgb(255, 192, 192)
        } else if elasticity > 1.0 {
            Color32::from_rgb(255, 222, 140)
        } else {
            Color32::from_rgb(172, 238, 190)
        }
    }

    /// Returns the high-contrast focus outline color (`>= 4.5:1`) for the viewport split sash.
    #[inline]
    #[must_use]
    pub const fn sash_focus_outline(visuals: &egui::Visuals) -> Color32 {
        if visuals.dark_mode {
            Color32::from_rgb(255, 214, 10)
        } else {
            Color32::from_rgb(0, 72, 186)
        }
    }
}
