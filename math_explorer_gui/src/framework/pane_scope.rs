#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{AccessibleHoverText, BrushMode, BrushStroke, InteractiveTool, PaneSidePanel};
use eframe::egui;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputMode {
    Mouse,
    Touch,
}

/// Event context provided to interaction hooks.
pub struct InteractionContext<'a> {
    /// Current pointer coordinates in screen space, if hovering or interacting.
    pub pointer_pos: Option<egui::Pos2>,
    /// Pointer drag delta since the previous frame.
    pub delta: egui::Vec2,
    /// Whether an active pointer drag gesture is underway.
    pub is_dragging: bool,
    /// Whether the canvas was primary-clicked this frame.
    pub is_clicked: bool,
    /// Underlying `egui` widget response for the canvas region.
    pub response: &'a egui::Response,
    /// Active pointer or touch input mode.
    pub input_mode: InputMode,
    /// Multi-touch pinch/pan gesture metadata when present.
    pub multi_touch: Option<egui::MultiTouchInfo>,
    /// Currently pressed keyboard keys during this frame.
    pub keys_down: std::collections::HashSet<egui::Key>,
    /// Active keyboard modifiers (`Ctrl`, `Shift`, `Alt`, `Command`).
    pub modifiers: egui::Modifiers,
    /// Optional interpolated spatial brush stroke emitted during drag sculpting.
    pub brush_stroke: Option<BrushStroke>,
}

impl<'a> InteractionContext<'a> {
    /// Builds an [`InteractionContext`] carrying an interpolated [`BrushStroke`] for [`InteractiveTool::on_brush`].
    #[must_use]
    pub fn from_brush_stroke(response: &'a egui::Response, stroke: BrushStroke) -> Self {
        Self {
            pointer_pos: response.interact_pointer_pos(),
            delta: response.drag_delta(),
            is_dragging: response.dragged(),
            is_clicked: response.clicked(),
            response,
            input_mode: InputMode::Mouse,
            multi_touch: None,
            keys_down: std::collections::HashSet::new(),
            modifiers: egui::Modifiers::NONE,
            brush_stroke: Some(stroke),
        }
    }
}

/// Computes a pane-salted panel [`egui::Id`] from the enclosing `ui` hierarchy.
#[inline]
pub fn scoped_panel_id(ui: &egui::Ui, suffix: impl std::hash::Hash) -> egui::Id {
    ui.id().with(suffix)
}

/// Returns `true` when unfocused pane-level keyboard shortcuts (e.g. `.`, `Ctrl+Z`) may execute in `ui`.
pub fn is_pane_shortcut_allowed(ui: &egui::Ui) -> bool {
    let ctx = ui.ctx();
    if ctx.wants_keyboard_input()
        || egui::Popup::is_any_open(ctx)
        || ctx.memory(|m| m.top_modal_layer().is_some())
    {
        return false;
    }
    ctx.data(|d| {
        if d.get_temp::<bool>(egui::Id::new("SHELL_MODAL_OR_SASH_FOCUSED"))
            .unwrap_or(false)
        {
            return false;
        }
        if let Some(rendering_pane) = d.get_temp::<usize>(egui::Id::new("CURRENT_RENDERING_PANE")) {
            let active_pane = d
                .get_temp::<usize>(egui::Id::new("ACTIVE_FOCUSED_PANE"))
                .unwrap_or(0);
            if rendering_pane != active_pane {
                return false;
            }
        }
        true
    })
}

/// Returns `true` when unfocused activation-key shortcuts (`Space` / `Enter`) may execute without colliding with focused widgets.
pub fn is_pane_activation_shortcut_allowed(ui: &egui::Ui) -> bool {
    is_pane_shortcut_allowed(ui) && ui.ctx().memory(|m| m.focused().is_none())
}

pub(super) fn render_default_tool_inside<T: InteractiveTool + ?Sized>(
    tool: &mut T,
    ui: &mut egui::Ui,
) {
    let panel_id = scoped_panel_id(ui, format!("{}_controls", tool.name()));
    PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
        ui.horizontal(|ui| {
            ui.heading(tool.name());
            let help_btn = ui
                .add(egui::Button::new("❓").small())
                .accessible_hover_text(format!("Toggle Theory Context Portal for {}", tool.name()));
            if help_btn.clicked() {
                let ctx = ui.ctx();
                let next = !ctx.data(|d| {
                    d.get_temp::<bool>(egui::Id::new("SHOW_THEORY_PORTAL"))
                        .unwrap_or(false)
                });
                ctx.data_mut(|d| {
                    d.insert_temp(egui::Id::new("SHOW_THEORY_PORTAL"), next);
                    if let Some(inst_key) =
                        d.get_temp::<egui::Id>(egui::Id::new("CURRENT_THEORY_PORTAL_INST_ID"))
                    {
                        d.insert_temp(inst_key, next);
                    }
                });
            }
        });
        ui.separator();
        tool.show_ui(ui);
    });

    render_default_tool_canvas(tool, ui);
}

fn build_default_brush_stroke(
    pointer_pos: Option<egui::Pos2>,
    delta: egui::Vec2,
    is_dragging: bool,
    secondary_down: bool,
) -> Option<BrushStroke> {
    let pos = pointer_pos.filter(|_| is_dragging)?;
    Some(BrushStroke {
        start: [f64::from(pos.x - delta.x), f64::from(pos.y - delta.y)],
        end: [f64::from(pos.x), f64::from(pos.y)],
        radius: 2.0,
        mode: if secondary_down {
            BrushMode::Erase
        } else {
            BrushMode::Solid
        },
        drag_vector: [f64::from(delta.x), f64::from(delta.y)],
    })
}

fn render_default_tool_canvas<T: InteractiveTool + ?Sized>(tool: &mut T, ui: &mut egui::Ui) {
    egui::CentralPanel::default().show_inside(ui, |ui| {
        let (response, painter) =
            ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
        let multi_touch = ui.input(|i| i.multi_touch());
        let input_mode = ui.ctx().data(|d| {
            d.get_temp(egui::Id::new("INPUT_MODE"))
                .unwrap_or(InputMode::Mouse)
        });
        let (keys_down, modifiers, secondary_down) =
            ui.input(|i| (i.keys_down.clone(), i.modifiers, i.pointer.secondary_down()));
        let pointer_pos = response.interact_pointer_pos();
        let delta = response.drag_delta();
        let is_dragging = response.dragged();

        let interaction_ctx = InteractionContext {
            pointer_pos,
            delta,
            is_dragging,
            is_clicked: response.clicked(),
            response: &response,
            input_mode,
            multi_touch,
            keys_down,
            modifiers,
            brush_stroke: build_default_brush_stroke(
                pointer_pos,
                delta,
                is_dragging,
                secondary_down,
            ),
        };

        if interaction_ctx.response.has_focus() {
            tool.on_keyboard(&interaction_ctx);
        }

        if interaction_ctx.multi_touch.is_some() {
            tool.on_gesture(&interaction_ctx);
        } else if interaction_ctx.pointer_pos.is_some() {
            if interaction_ctx.is_dragging {
                tool.on_drag(&interaction_ctx);
                tool.on_brush(&interaction_ctx);
            } else if interaction_ctx.is_clicked {
                tool.on_click(&interaction_ctx);
            } else {
                tool.on_hover(&interaction_ctx);
            }
        }

        tool.draw(ui, &response, &painter);
    });
}
