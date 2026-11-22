#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::stencils::{BrushStencil, SymmetryMode};
use eframe::egui;

/// Persistent UI state for [`super::RasterBrushCanvas`], tracking brush configuration, pointer drag, and WCAG reticle.
#[derive(Clone, Debug)]
pub struct RasterBrushState {
    /// Brush stamp radius in discrete cells (`0..=max_radius`).
    pub radius: usize,
    /// Active geometric brush or obstacle stencil.
    pub stencil: BrushStencil,
    /// Active mirror/radial symmetry mode.
    pub symmetry: SymmetryMode,
    /// Keyboard reticle coordinates `(x, y)` in solver/grid space.
    pub reticle: (usize, usize),
    pub(super) reticle_initialized: bool,
    pub(super) drag_prev: Option<(usize, usize)>,
    pub(super) drag_secondary: bool,
    /// Active start endpoint when [`BrushStencil::Line`] is anchored.
    pub line_anchor: Option<(usize, usize)>,
    /// Last rendered `egui::Id` when focused.
    pub focused_id: Option<egui::Id>,
}

impl Default for RasterBrushState {
    fn default() -> Self {
        Self {
            radius: 2,
            stencil: BrushStencil::Freehand,
            symmetry: SymmetryMode::None,
            reticle: (0, 0),
            reticle_initialized: false,
            drag_prev: None,
            drag_secondary: false,
            line_anchor: None,
            focused_id: None,
        }
    }
}

impl RasterBrushState {
    /// Creates a [`RasterBrushState`] initialized with a custom brush `radius`.
    #[must_use]
    pub fn with_radius(radius: usize) -> Self {
        Self {
            radius,
            ..Self::default()
        }
    }

    /// Creates a [`RasterBrushState`] initialized with a custom `stencil` and `radius`.
    #[must_use]
    pub fn with_stencil_and_radius(stencil: BrushStencil, radius: usize) -> Self {
        Self {
            radius,
            stencil,
            ..Self::default()
        }
    }

    /// Ensures `reticle` is initialized near grid center and clamped to `(grid_w, grid_h)`.
    pub fn clamp_to_grid(&mut self, grid_w: usize, grid_h: usize) {
        if grid_w == 0 || grid_h == 0 {
            self.reticle = (0, 0);
            return;
        }
        if !self.reticle_initialized {
            self.reticle = (grid_w / 2, grid_h / 2);
            self.reticle_initialized = true;
        } else {
            self.reticle.0 = self.reticle.0.min(grid_w - 1);
            self.reticle.1 = self.reticle.1.min(grid_h - 1);
        }
        if let Some((ax, ay)) = self.line_anchor {
            if ax >= grid_w || ay >= grid_h {
                self.line_anchor = None;
            }
        }
    }
}

/// Discrete cell batch emitted by [`super::RasterBrushCanvas`] from pointer strokes or keyboard reticle actions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RasterBrushStroke {
    /// Unique in-bounds `(x, y)` grid coordinates touched by this stroke or stamp.
    pub cells: Vec<(usize, usize)>,
    /// `true` when triggered by secondary pointer button or `Backspace`/`Delete` erase key.
    pub secondary: bool,
    /// `true` when triggered via keyboard `Space` / `Backspace` / `Delete`.
    pub via_keyboard: bool,
    /// Primary target cell `(x, y)` where the stroke landed.
    pub focus_cell: (usize, usize),
}
