//! Reusable spatial brush controller (`SpatialBrushKit`) for 2D raster grids and 1D/2D
//! `egui_plot` canvases with WCAG 2.2 AA keyboard sculpting and focus indicators.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::announce_live_status;
use eframe::egui::{self, EventFilter, Key, Modifiers, Pos2, Rect, Shape};
use egui_plot::{
    commands::{CommandRegistryData, CommandTrigger},
    focus_ring_stroke, PlotPoint, PlotUi,
};

/// Controls whether pointer/keyboard gestures pan/zoom the viewport or sculpt spatial fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum BrushInteractionMode {
    /// Standard plot navigation (drag pan / scroll zoom).
    Pan,
    /// Active field sculpting via pointer drag and keyboard cursor.
    #[default]
    Sculpt,
}

/// Semantic brush action dispatched across fluid, reaction-diffusion, and quantum tools.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum BrushMode {
    /// Solid bounce-back obstacle (LBM).
    #[default]
    Solid,
    /// Clears obstacles, cell masks, or potential deltas back to baseline.
    Erase,
    /// Injects directed fluid momentum along the brush drag vector (LBM).
    VelocityJet,
    /// Impermeable internal zero-flux Neumann barrier (Morphogenesis).
    ZeroFluxWall,
    /// Persistent activator species `u` emitter (Morphogenesis).
    EmitterU,
    /// Persistent inhibitor species `v` emitter (Morphogenesis).
    EmitterV,
    /// Chemical absorber clamping concentrations to zero (Morphogenesis).
    Sink,
    /// Positive finite potential barrier $V(x) > 0$ (Quantum).
    PotentialBarrier,
    /// Negative potential well $V(x) < 0$ (Quantum).
    PotentialWell,
    /// Step potential plateau from cursor $x$ to the right boundary (Quantum).
    PotentialStep,
    /// Places or steers a potential-flow source/sink singularity.
    FlowSource,
    /// Places or steers a potential-flow point vortex singularity.
    FlowVortex,
    /// Places or steers a potential-flow doublet singularity.
    FlowDoublet,
}

impl BrushMode {
    /// Human-readable label announced to assistive technologies.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Solid => "Solid Wall",
            Self::Erase => "Erase",
            Self::VelocityJet => "Velocity Jet",
            Self::ZeroFluxWall => "Zero-Flux Wall",
            Self::EmitterU => "u Emitter",
            Self::EmitterV => "v Emitter",
            Self::Sink => "Chemical Sink",
            Self::PotentialBarrier => "Finite Barrier",
            Self::PotentialWell => "Potential Well",
            Self::PotentialStep => "Potential Step",
            Self::FlowSource => "Source / Sink",
            Self::FlowVortex => "Vortex",
            Self::FlowDoublet => "Doublet",
        }
    }
}

/// Continuous brush segment in plot coordinates dispatched to [`super::InteractiveTool::on_brush`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BrushStroke {
    /// Segment start position `[x, y]` in plot coordinates.
    pub start: [f64; 2],
    /// Segment end position `[x, y]` in plot coordinates.
    pub end: [f64; 2],
    /// Brush radius in plot units.
    pub radius: f64,
    /// Active brush operation mode.
    pub mode: BrushMode,
    /// Pointer displacement `[dx, dy]` in plot coordinates across this frame.
    pub drag_vector: [f64; 2],
}

/// Stateful controller sampling pointer drags and WCAG 2.2 AA keyboard sculpting inside `egui_plot`.
#[derive(Debug, Clone)]
pub struct SpatialBrushKit {
    /// Whether pointer/keyboard gestures pan or sculpt.
    pub interaction_mode: BrushInteractionMode,
    /// Active primary brush mode.
    pub mode: BrushMode,
    /// Current brush radius in plot coordinates.
    pub radius: f64,
    /// Minimum selectable brush radius.
    pub min_radius: f64,
    /// Maximum selectable brush radius.
    pub max_radius: f64,
    /// Radius increment per `[` / `]` keypress.
    pub radius_step: f64,
    /// Current keyboard/reticle cursor position in plot coordinates.
    pub cursor_pos: [f64; 2],
    /// Arrow-key cursor step `[dx, dy]` in plot coordinates.
    pub cursor_step: [f64; 2],
    cursor_initialized: bool,
    last_drag_pos: Option<[f64; 2]>,
    last_focused_id: Option<egui::Id>,
}

impl Default for SpatialBrushKit {
    fn default() -> Self {
        Self::new(BrushMode::Solid, 2.0)
    }
}

impl SpatialBrushKit {
    /// Creates a new `SpatialBrushKit` with `default_mode` and initial `radius`.
    #[must_use]
    pub fn new(default_mode: BrushMode, radius: f64) -> Self {
        Self {
            interaction_mode: BrushInteractionMode::Sculpt,
            mode: default_mode,
            radius: radius.max(0.05),
            min_radius: 1.0,
            max_radius: 12.0,
            radius_step: 1.0,
            cursor_pos: [0.0, 0.0],
            cursor_step: [2.0, 2.0],
            cursor_initialized: false,
            last_drag_pos: None,
            last_focused_id: None,
        }
    }

    /// Configures radius bounds and keyboard step increment.
    #[must_use]
    pub fn with_radius_bounds(mut self, min_r: f64, max_r: f64, step: f64) -> Self {
        self.min_radius = min_r.max(0.05);
        self.max_radius = max_r.max(self.min_radius);
        self.radius_step = step.max(0.05);
        self.radius = self.radius.clamp(self.min_radius, self.max_radius);
        self
    }

    /// Configures initial [`BrushInteractionMode`].
    #[must_use]
    pub fn with_interaction_mode(mut self, mode: BrushInteractionMode) -> Self {
        self.interaction_mode = mode;
        self
    }

    /// Configures arrow-key step sizes `[dx, dy]` in plot coordinates.
    #[must_use]
    pub fn with_cursor_step(mut self, dx: f64, dy: f64) -> Self {
        self.cursor_step = [dx.abs().max(1e-3), dy.abs().max(1e-3)];
        self
    }

    /// Returns `true` when [`BrushInteractionMode::Sculpt`] is active.
    #[must_use]
    pub fn is_sculpting(&self) -> bool {
        self.interaction_mode == BrushInteractionMode::Sculpt
    }

    /// Returns the most recently focused plot canvas [`egui::Id`], if currently focused.
    #[must_use]
    pub fn focused_plot_id(&self) -> Option<egui::Id> {
        self.last_focused_id
    }

    /// Maps plot coordinates `(x, y)` to a solver grid cell where solver `y = 0` is at the bottom
    /// (`PlotImage` row `height - 1 - y`, as used by `LbmUnified`).
    #[must_use]
    pub fn plot_to_grid_y_up(pt: [f64; 2], width: usize, height: usize) -> (i32, i32) {
        let max_x = width.saturating_sub(1) as i32;
        let max_y = height.saturating_sub(1) as i32;
        let gx = (pt[0].floor() as i32).clamp(0, max_x);
        let gy = (pt[1].floor() as i32).clamp(0, max_y);
        (gx, gy)
    }

    /// Maps plot coordinates `(x, y)` to a solver grid cell where solver `y = 0` is at the top
    /// (`PlotImage` row `y`, as used by `MorphogenesisTool`).
    #[must_use]
    pub fn plot_to_grid_y_down(pt: [f64; 2], width: usize, height: usize) -> (i32, i32) {
        let max_x = width.saturating_sub(1) as i32;
        let max_y = height.saturating_sub(1) as i32;
        let gx = (pt[0].floor() as i32).clamp(0, max_x);
        let plot_row = (pt[1].floor() as i32).clamp(0, max_y);
        let gy = (max_y - plot_row).clamp(0, max_y);
        (gx, gy)
    }

    /// Samples pointer drags and keyboard sculpting inside `plot_ui`, renders high-contrast
    /// focus/cursor overlays, and returns any `BrushStroke`s generated on this frame.
    pub fn sample_plot(
        &mut self,
        plot_ui: &mut PlotUi<'_>,
        domain_min: [f64; 2],
        domain_max: [f64; 2],
        secondary_mode: Option<BrushMode>,
    ) -> Vec<BrushStroke> {
        self.ensure_cursor_in_bounds(domain_min, domain_max);
        let ctx = plot_ui.ctx().clone();
        Self::register_shortcuts(&ctx);

        let response = plot_ui.response().clone();
        if response.has_focus() {
            self.last_focused_id = Some(response.id);
        } else if self.last_focused_id == Some(response.id) {
            self.last_focused_id = None;
        }

        let mut strokes = Vec::new();
        if self.is_sculpting() {
            if response.has_focus() {
                self.sample_keyboard(&ctx, response.id, domain_min, domain_max, &mut strokes);
            }
            self.sample_pointer(plot_ui, &ctx, &response, secondary_mode, &mut strokes);
        } else {
            self.last_drag_pos = None;
        }

        self.draw_focus_and_reticle(plot_ui, &ctx, &response);
        strokes
    }

    fn ensure_cursor_in_bounds(&mut self, min: [f64; 2], max: [f64; 2]) {
        let lo_x = min[0].min(max[0]);
        let hi_x = min[0].max(max[0]);
        let lo_y = min[1].min(max[1]);
        let hi_y = min[1].max(max[1]);
        if !self.cursor_initialized {
            self.cursor_pos = [0.5 * (lo_x + hi_x), 0.5 * (lo_y + hi_y)];
            self.cursor_initialized = true;
        } else {
            self.cursor_pos[0] = self.cursor_pos[0].clamp(lo_x, hi_x);
            self.cursor_pos[1] = self.cursor_pos[1].clamp(lo_y, hi_y);
        }
    }

    fn register_shortcuts(ctx: &egui::Context) {
        CommandRegistryData::register(
            ctx,
            "Arrow Keys",
            "Move spatial brush cursor on focused canvas",
            CommandTrigger::Key(Key::ArrowRight),
            false,
            "Spatial Brush",
        );
        CommandRegistryData::register(
            ctx,
            "[ / ]",
            "Decrease or increase spatial brush radius",
            CommandTrigger::Key(Key::OpenBracket),
            false,
            "Spatial Brush",
        );
        CommandRegistryData::register(
            ctx,
            "Space / Enter",
            "Stamp spatial brush at keyboard cursor",
            CommandTrigger::Key(Key::Space),
            false,
            "Spatial Brush",
        );
    }

    fn consume_keyboard_input(
        &mut self,
        i: &mut egui::InputState,
        domain_min: [f64; 2],
        domain_max: [f64; 2],
        strokes: &mut Vec<BrushStroke>,
    ) -> Option<String> {
        if i.modifiers.ctrl || i.modifiers.command || i.modifiers.alt {
            return None;
        }
        let mut announcement = None;
        let mut moved = false;
        for (key, dx, dy) in [
            (Key::ArrowLeft, -self.cursor_step[0], 0.0),
            (Key::ArrowRight, self.cursor_step[0], 0.0),
            (Key::ArrowUp, 0.0, self.cursor_step[1]),
            (Key::ArrowDown, 0.0, -self.cursor_step[1]),
        ] {
            if i.consume_key(Modifiers::SHIFT, key) {
                self.cursor_pos[0] += 3.0 * dx;
                self.cursor_pos[1] += 3.0 * dy;
                moved = true;
            } else if i.consume_key(Modifiers::NONE, key) {
                self.cursor_pos[0] += dx;
                self.cursor_pos[1] += dy;
                moved = true;
            }
        }
        if moved {
            self.ensure_cursor_in_bounds(domain_min, domain_max);
            announcement = Some(format!(
                "Brush cursor at ({:.1}, {:.1}), mode {}",
                self.cursor_pos[0],
                self.cursor_pos[1],
                self.mode.label()
            ));
        }
        if i.consume_key(Modifiers::NONE, Key::OpenBracket) {
            self.radius = (self.radius - self.radius_step).clamp(self.min_radius, self.max_radius);
            announcement = Some(format!("Brush radius {:.1}", self.radius));
        }
        if i.consume_key(Modifiers::NONE, Key::CloseBracket) {
            self.radius = (self.radius + self.radius_step).clamp(self.min_radius, self.max_radius);
            announcement = Some(format!("Brush radius {:.1}", self.radius));
        }
        if i.consume_key(Modifiers::NONE, Key::Space) || i.consume_key(Modifiers::NONE, Key::Enter)
        {
            strokes.push(BrushStroke {
                start: self.cursor_pos,
                end: self.cursor_pos,
                radius: self.radius,
                mode: self.mode,
                drag_vector: [0.0, 0.0],
            });
            announcement = Some(format!(
                "Stamped {} at ({:.1}, {:.1}) radius {:.1}",
                self.mode.label(),
                self.cursor_pos[0],
                self.cursor_pos[1],
                self.radius
            ));
        }
        announcement
    }

    fn sample_keyboard(
        &mut self,
        ctx: &egui::Context,
        plot_id: egui::Id,
        domain_min: [f64; 2],
        domain_max: [f64; 2],
        strokes: &mut Vec<BrushStroke>,
    ) {
        // Lock ONLY arrow keys while focused; never trap Tab, Shift+Tab, or Escape.
        ctx.memory_mut(|m| {
            m.set_focus_lock_filter(
                plot_id,
                EventFilter {
                    tab: false,
                    horizontal_arrows: true,
                    vertical_arrows: true,
                    escape: false,
                },
            );
        });

        let announcement =
            ctx.input_mut(|i| self.consume_keyboard_input(i, domain_min, domain_max, strokes));

        if let Some(msg) = announcement {
            announce_live_status(ctx, &msg);
            ctx.request_repaint();
        }
    }

    fn sample_pointer(
        &mut self,
        plot_ui: &PlotUi<'_>,
        ctx: &egui::Context,
        response: &egui::Response,
        secondary_mode: Option<BrushMode>,
        strokes: &mut Vec<BrushStroke>,
    ) {
        let (primary, secondary) =
            ctx.input(|i| (i.pointer.primary_down(), i.pointer.secondary_down()));
        let active_mode = if primary {
            Some(self.mode)
        } else if secondary {
            secondary_mode
        } else {
            None
        };

        let Some(mode) = active_mode else {
            self.last_drag_pos = None;
            return;
        };

        let pointer_in_plot = response.hovered() || response.dragged();
        if !pointer_in_plot {
            self.last_drag_pos = None;
            return;
        }

        if let Some(coord) = plot_ui.pointer_coordinate() {
            let curr = [coord.x, coord.y];
            let prev = self.last_drag_pos.unwrap_or(curr);
            let drag_vector = [curr[0] - prev[0], curr[1] - prev[1]];
            self.cursor_pos = curr;
            self.cursor_initialized = true;
            self.last_drag_pos = Some(curr);
            strokes.push(BrushStroke {
                start: prev,
                end: curr,
                radius: self.radius,
                mode,
                drag_vector,
            });
        }
    }

    fn draw_focus_and_reticle(
        &self,
        plot_ui: &PlotUi<'_>,
        ctx: &egui::Context,
        response: &egui::Response,
    ) {
        let has_focus = response.has_focus();
        let show_reticle = self.is_sculpting() && (has_focus || response.hovered());
        if !has_focus && !show_reticle {
            return;
        }

        let visuals = ctx.style().visuals.clone();
        let stroke = focus_ring_stroke(&visuals, visuals.extreme_bg_color);
        let transform = plot_ui.transform();
        let frame: Rect = *transform.frame();
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            response.id.with("spatial_brush_overlay"),
        ));

        if has_focus {
            painter.rect_stroke(frame.shrink(1.0), 2.0, stroke, egui::StrokeKind::Inside);
        }

        if show_reticle {
            let center_pt = PlotPoint::new(self.cursor_pos[0], self.cursor_pos[1]);
            let edge_pt = PlotPoint::new(self.cursor_pos[0] + self.radius, self.cursor_pos[1]);
            let center_px: Pos2 = transform.position_from_point(&center_pt);
            let edge_px: Pos2 = transform.position_from_point(&edge_pt);
            let px_radius = (edge_px.x - center_px.x).abs().max(4.0);
            let clipped = painter.with_clip_rect(frame);
            clipped.add(Shape::circle_stroke(center_px, px_radius, stroke));
            let arm = 4.0_f32;
            clipped.line_segment(
                [
                    Pos2::new(center_px.x - arm, center_px.y),
                    Pos2::new(center_px.x + arm, center_px.y),
                ],
                stroke,
            );
            clipped.line_segment(
                [
                    Pos2::new(center_px.x, center_px.y - arm),
                    Pos2::new(center_px.x, center_px.y + arm),
                ],
                stroke,
            );
        }
    }
}
