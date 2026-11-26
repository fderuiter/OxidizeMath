//! Composable plot brush controller bridging `egui_plot` canvases to [`super::InteractiveTool`]
//! normalized hooks (`on_keyboard`, `on_drag`, `on_brush`, `on_click`, `on_hover`) and
//! gap-free integer Bresenham [`crate::async_sim::SimCommand::ApplyBrush`] strokes.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::{lock_canvas_arrow_keys, paint_canvas_focus_ring};
use crate::async_sim::SimCommand;
use crate::framework::{
    BrushMode, BrushStroke, InputMode, InteractionContext, InteractiveTool, SpatialBrushKit,
};
use eframe::egui::{self, Key, Modifiers, Pos2, Vec2};
use egui_plot::PlotUi;
use std::collections::HashSet;

/// Per-frame interaction sample captured inside `egui_plot::Plot::show` and dispatched
/// to [`InteractiveTool`] hooks once the plot borrow closes.
#[derive(Debug, Clone)]
pub struct PlotBrushFrame {
    /// Interpolated plot-space brush strokes generated via pointer drag or keyboard stamp.
    pub strokes: Vec<BrushStroke>,
    /// Pointer coordinates `[x, y]` in plot space, if available.
    pub plot_pointer_pos: Option<[f64; 2]>,
    /// Pointer delta `[dx, dy]` in plot space across the current frame.
    pub plot_delta: [f64; 2],
    /// Screen-space pointer delta reported by `egui`.
    pub screen_delta: Vec2,
    /// Whether an active pointer drag occurred on the plot canvas.
    pub is_dragging: bool,
    /// Whether a primary or secondary click occurred on the plot canvas.
    pub is_clicked: bool,
    /// Whether the pointer hovered over the plot canvas.
    pub is_hovered: bool,
    /// Whether the plot canvas currently holds keyboard focus.
    pub has_focus: bool,
    /// Active input modality (`Mouse` vs. `Touch`).
    pub input_mode: InputMode,
    /// Active multi-touch gesture state, if any.
    pub multi_touch: Option<egui::MultiTouchInfo>,
    /// Currently pressed keyboard keys.
    pub keys_down: HashSet<Key>,
    /// Active keyboard modifiers (`Ctrl`, `Shift`, `Alt`, `Cmd`).
    pub modifiers: Modifiers,
}

impl PlotBrushFrame {
    fn plot_pos2(&self) -> Option<Pos2> {
        self.plot_pointer_pos
            .map(|[x, y]| Pos2::new(x as f32, y as f32))
    }

    fn plot_vec2(&self) -> Vec2 {
        if self.plot_delta[0].abs() > 0.0 || self.plot_delta[1].abs() > 0.0 {
            Vec2::new(self.plot_delta[0] as f32, self.plot_delta[1] as f32)
        } else {
            self.screen_delta
        }
    }

    fn build_context<'a>(
        &self,
        response: &'a egui::Response,
        brush_stroke: Option<BrushStroke>,
    ) -> InteractionContext<'a> {
        let pointer_pos = brush_stroke
            .map(|s| Pos2::new(s.end[0] as f32, s.end[1] as f32))
            .or_else(|| self.plot_pos2());
        let delta = brush_stroke.map_or_else(
            || self.plot_vec2(),
            |s| Vec2::new(s.drag_vector[0] as f32, s.drag_vector[1] as f32),
        );
        InteractionContext {
            pointer_pos,
            delta,
            is_dragging: self.is_dragging,
            is_clicked: self.is_clicked,
            response,
            input_mode: self.input_mode,
            multi_touch: self.multi_touch,
            keys_down: self.keys_down.clone(),
            modifiers: self.modifiers,
            brush_stroke,
        }
    }

    /// Applies WCAG 2.2 AA canvas focus affordances and dispatches normalized
    /// [`InteractiveTool`] hooks (`on_keyboard`, `on_gesture`, `on_drag`, `on_click`,
    /// `on_hover`, and `on_brush`).
    pub fn dispatch<T: InteractiveTool + ?Sized>(
        &self,
        ui: &egui::Ui,
        response: &egui::Response,
        tool: &mut T,
    ) {
        if self.has_focus {
            paint_canvas_focus_ring(ui, response);
            lock_canvas_arrow_keys(ui, response);
        }

        let base_ctx = self.build_context(response, self.strokes.first().copied());
        if self.has_focus {
            tool.on_keyboard(&base_ctx);
        }

        if self.multi_touch.is_some() {
            tool.on_gesture(&base_ctx);
        } else if self.is_dragging {
            tool.on_drag(&base_ctx);
        } else if self.is_clicked {
            tool.on_click(&base_ctx);
        } else if self.is_hovered && base_ctx.pointer_pos.is_some() {
            tool.on_hover(&base_ctx);
        }

        for &stroke in &self.strokes {
            let stroke_ctx = self.build_context(response, Some(stroke));
            tool.on_brush(&stroke_ctx);
        }
    }

    /// Converts all captured [`BrushStroke`]s into integer Bresenham [`SimCommand::ApplyBrush`] commands.
    #[must_use]
    pub fn bresenham_commands(&self, width: usize, height: usize, y_up: bool) -> Vec<SimCommand> {
        let mut cmds = Vec::new();
        for stroke in &self.strokes {
            PlotBrushController::append_bresenham_commands(stroke, width, height, y_up, &mut cmds);
        }
        cmds
    }
}

/// Composable controller bridging custom `egui_plot` views to [`SpatialBrushKit`],
/// [`InteractiveTool`] hooks, and zero-allocation integer Bresenham strokes.
#[derive(Debug, Clone, Default)]
pub struct PlotBrushController {
    /// Underlying spatial brush cursor, mode, and radius state.
    pub kit: SpatialBrushKit,
}

impl PlotBrushController {
    /// Creates a new `PlotBrushController` with `default_mode` and initial `radius`.
    #[must_use]
    pub fn new(default_mode: BrushMode, radius: f64) -> Self {
        Self {
            kit: SpatialBrushKit::new(default_mode, radius),
        }
    }

    /// Wraps an existing [`SpatialBrushKit`] configuration.
    #[must_use]
    pub fn from_kit(kit: SpatialBrushKit) -> Self {
        Self { kit }
    }

    /// Returns the focused plot canvas [`egui::Id`], if currently focused.
    #[must_use]
    pub fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.kit.focused_plot_id()
    }

    /// Samples `plot_ui` using this controller's internal [`SpatialBrushKit`].
    pub fn sample_plot(
        &mut self,
        plot_ui: &mut PlotUi<'_>,
        domain_min: [f64; 2],
        domain_max: [f64; 2],
        secondary_mode: Option<BrushMode>,
    ) -> PlotBrushFrame {
        Self::capture_plot(
            &mut self.kit,
            plot_ui,
            domain_min,
            domain_max,
            secondary_mode,
        )
    }

    /// Captures pointer, touch, focus, and keyboard brush state from `plot_ui` using `kit`.
    pub fn capture_plot(
        kit: &mut SpatialBrushKit,
        plot_ui: &mut PlotUi<'_>,
        domain_min: [f64; 2],
        domain_max: [f64; 2],
        secondary_mode: Option<BrushMode>,
    ) -> PlotBrushFrame {
        let strokes = kit.sample_plot(plot_ui, domain_min, domain_max, secondary_mode);
        let ctx = plot_ui.ctx().clone();
        let resp = plot_ui.response().clone();
        let plot_pointer_pos = plot_ui.pointer_coordinate().map(|p| [p.x, p.y]);
        let plot_delta = {
            let d = plot_ui.pointer_coordinate_drag_delta();
            [f64::from(d.x), f64::from(d.y)]
        };
        let input_mode = ctx.data(|d| {
            d.get_temp(egui::Id::new("INPUT_MODE"))
                .unwrap_or(InputMode::Mouse)
        });
        let (multi_touch, keys_down, modifiers, screen_delta, primary_down, secondary_down) = ctx
            .input(|i| {
                (
                    i.multi_touch(),
                    i.keys_down.clone(),
                    i.modifiers,
                    i.pointer.delta(),
                    i.pointer.primary_down(),
                    i.pointer.secondary_down(),
                )
            });
        let active_drag = resp.dragged()
            || ((resp.hovered() || !strokes.is_empty()) && (primary_down || secondary_down));
        let is_clicked = resp.clicked() || resp.secondary_clicked();

        PlotBrushFrame {
            strokes,
            plot_pointer_pos,
            plot_delta,
            screen_delta,
            is_dragging: active_drag,
            is_clicked,
            is_hovered: resp.hovered(),
            has_focus: resp.has_focus(),
            input_mode,
            multi_touch,
            keys_down,
            modifiers,
        }
    }

    /// Walks an integer Bresenham line between `stroke.start` and `stroke.end` with zero heap
    /// allocations, invoking `visit(cx, cy, radius, is_obstacle)` at every lattice cell.
    pub fn for_each_bresenham_step(
        stroke: &BrushStroke,
        width: usize,
        height: usize,
        y_up: bool,
        mut visit: impl FnMut(i32, i32, i32, bool),
    ) {
        if width == 0 || height == 0 {
            return;
        }
        let map_pt = if y_up {
            SpatialBrushKit::plot_to_grid_y_up
        } else {
            SpatialBrushKit::plot_to_grid_y_down
        };
        let (mut x, mut y) = map_pt(stroke.start, width, height);
        let (x1, y1) = map_pt(stroke.end, width, height);
        let radius = (stroke.radius.round() as i32).clamp(1, 8);
        let is_obstacle = stroke.mode != BrushMode::Erase;

        let dx = (x1 - x).abs();
        let sx = if x < x1 { 1 } else { -1 };
        let dy = -(y1 - y).abs();
        let sy = if y < y1 { 1 } else { -1 };
        let mut err = dx + dy;

        loop {
            visit(x, y, radius, is_obstacle);
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// Appends gap-free integer Bresenham [`SimCommand::ApplyBrush`] steps for `stroke` into `out`.
    pub fn append_bresenham_commands(
        stroke: &BrushStroke,
        width: usize,
        height: usize,
        y_up: bool,
        out: &mut Vec<SimCommand>,
    ) {
        Self::for_each_bresenham_step(stroke, width, height, y_up, |cx, cy, r, is_obstacle| {
            out.push(SimCommand::ApplyBrush {
                cx,
                cy,
                r,
                is_obstacle,
            });
        });
    }

    /// Returns gap-free integer Bresenham [`SimCommand::ApplyBrush`] steps for `stroke`.
    #[must_use]
    pub fn bresenham_sim_commands(
        stroke: &BrushStroke,
        width: usize,
        height: usize,
        y_up: bool,
    ) -> Vec<SimCommand> {
        let mut out = Vec::new();
        Self::append_bresenham_commands(stroke, width, height, y_up, &mut out);
        out
    }
}
