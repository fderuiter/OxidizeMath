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
    /// Four-state Ising spin fills `[+1 free, -1 free, +2 pinned, -2 pinned]` (`>= 3.0:1` polarity separation).
    pub ising_spins: [Color32; 4],
    /// Three-stop attention heatmap ramp `[low, mid, high]` (`>= 3.0:1` endpoint separation).
    pub attention_stops: [Color32; 3],
    /// Dark obstacle hatch/rim tone for solid boundaries.
    pub obstacle_dark: Color32,
    /// Light obstacle hatch/rim tone (`>= 3.0:1` against `obstacle_dark` and dark fluid).
    pub obstacle_light: Color32,
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
    ising_spins: [
        Color32::from_rgb(255, 184, 108),
        Color32::from_rgb(28, 72, 156),
        Color32::from_rgb(255, 232, 104),
        Color32::from_rgb(18, 96, 128),
    ],
    attention_stops: [
        Color32::from_rgb(24, 34, 68),
        Color32::from_rgb(64, 128, 204),
        Color32::from_rgb(250, 212, 84),
    ],
    obstacle_dark: Color32::from_rgb(20, 24, 34),
    obstacle_light: Color32::from_rgb(236, 240, 248),
};

/// Semantic palette tuned for [`ThemeMode::Light`] (classroom projection and print).
pub const LIGHT_PALETTE: SemanticPalette = SemanticPalette {
    info: Color32::from_rgb(0, 66, 138),
    success: Color32::from_rgb(15, 115, 54),
    warning: Color32::from_rgb(92, 46, 0),
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
    accent: Color32::from_rgb(96, 42, 0),
    series: [
        Color32::from_rgb(12, 88, 184),
        Color32::from_rgb(176, 68, 0),
        Color32::from_rgb(15, 115, 54),
        Color32::from_rgb(180, 28, 56),
        Color32::from_rgb(112, 42, 172),
        Color32::from_rgb(138, 90, 0),
    ],
    ising_spins: [
        Color32::from_rgb(255, 192, 124),
        Color32::from_rgb(22, 68, 152),
        Color32::from_rgb(255, 234, 112),
        Color32::from_rgb(12, 94, 122),
    ],
    attention_stops: [
        Color32::from_rgb(238, 243, 252),
        Color32::from_rgb(108, 152, 218),
        Color32::from_rgb(18, 48, 124),
    ],
    obstacle_dark: Color32::from_rgb(28, 34, 46),
    obstacle_light: Color32::from_rgb(246, 248, 252),
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
    ising_spins: [
        Color32::from_rgb(255, 202, 104),
        Color32::from_rgb(44, 88, 212),
        Color32::from_rgb(255, 244, 72),
        Color32::from_rgb(24, 108, 154),
    ],
    attention_stops: [
        Color32::from_rgb(52, 96, 196),
        Color32::from_rgb(140, 190, 250),
        Color32::from_rgb(255, 238, 56),
    ],
    obstacle_dark: Color32::BLACK,
    obstacle_light: Color32::from_rgb(255, 255, 255),
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

    /// Maps discrete Ising spin values (`+1`, `-1`, `+2` pinned up, `-2` pinned down) to
    /// luminance-separated semantic tokens (`>= 3.0:1` up-vs-down contrast across all themes).
    #[inline]
    #[must_use]
    pub const fn ising_spin_color(self, spin: i8) -> Color32 {
        match spin {
            2 => self.ising_spins[2],
            -2 => self.ising_spins[3],
            s if s >= 0 => self.ising_spins[0],
            _ => self.ising_spins[1],
        }
    }

    /// Evaluates the stack-allocated three-stop attention ramp for `weight` in `[0.0, 1.0]`.
    #[inline]
    #[must_use]
    pub fn attention_ramp_color(self, weight: f64) -> Color32 {
        let clamped = if weight.is_finite() {
            weight.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (a, b, t) = if clamped <= 0.5 {
            (
                self.attention_stops[0],
                self.attention_stops[1],
                clamped * 2.0,
            )
        } else {
            (
                self.attention_stops[1],
                self.attention_stops[2],
                (clamped - 0.5) * 2.0,
            )
        };
        let lerp_u8 = |lo: u8, hi: u8| -> u8 {
            let v = f64::from(lo) + (f64::from(hi) - f64::from(lo)) * t;
            v.round().clamp(0.0, 255.0) as u8
        };
        Color32::from_rgb(
            lerp_u8(a.r(), b.r()),
            lerp_u8(a.g(), b.g()),
            lerp_u8(a.b(), b.b()),
        )
    }

    /// Selects the obstacle rim tone (`obstacle_light` or `obstacle_dark`) maximizing WCAG 2.2
    /// contrast (`>= 3.5:1`) against an adjacent fluid fill.
    #[inline]
    #[must_use]
    pub fn obstacle_rim_color(self, adjacent_fluid: Color32) -> Color32 {
        let light_cr = super::theme::contrast_ratio(adjacent_fluid, self.obstacle_light);
        let dark_cr = super::theme::contrast_ratio(adjacent_fluid, self.obstacle_dark);
        if light_cr >= dark_cr {
            self.obstacle_light
        } else {
            self.obstacle_dark
        }
    }

    /// Returns the diagonal interior hatch color (`>= 14.9:1` internal stripe contrast) for an obstacle cell.
    #[inline]
    #[must_use]
    pub const fn obstacle_interior_hatch(self, x: usize, y: usize) -> Color32 {
        if (x + y).is_multiple_of(3) {
            self.obstacle_light
        } else {
            self.obstacle_dark
        }
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

    /// Returns the pill background fill for `tier` in `mode`, preserving `>= 4.5:1` WCAG 2.2 AA
    /// text contrast against [`Self::success`], [`Self::info`], and [`Self::warning`].
    #[inline]
    #[must_use]
    pub const fn maturity_badge_bg(
        mode: ThemeMode,
        tier: scientific_metadata::theory::MaturityTier,
    ) -> Color32 {
        use scientific_metadata::theory::MaturityTier;
        match (mode, tier) {
            (ThemeMode::Dark, MaturityTier::Certified) => Color32::from_rgb(14, 48, 26),
            (ThemeMode::Dark, MaturityTier::Validated) => Color32::from_rgb(12, 38, 66),
            (ThemeMode::Dark, MaturityTier::Experimental) => Color32::from_rgb(52, 36, 8),
            (ThemeMode::Light, MaturityTier::Certified) => Color32::from_rgb(232, 248, 237),
            (ThemeMode::Light, MaturityTier::Validated) => Color32::from_rgb(232, 243, 255),
            (ThemeMode::Light, MaturityTier::Experimental) => Color32::from_rgb(255, 246, 226),
            (ThemeMode::HighContrast, _) => Color32::BLACK,
        }
    }

    /// Converts an approved `math_commons::math_kernel::colormap` `[r, g, b]` triple into [`Color32`].
    #[inline]
    #[must_use]
    pub const fn colormap_to_color32(rgb: [u8; 3]) -> Color32 {
        Color32::from_rgb(rgb[0], rgb[1], rgb[2])
    }
}

#[cfg(test)]
#[path = "color_guard_tests.rs"]
pub(crate) mod color_guard_tests;

#[cfg(test)]
#[path = "canvas_overlay_tests.rs"]
mod canvas_overlay_tests;
