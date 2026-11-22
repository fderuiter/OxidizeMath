#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{AccessibleHoverText, InputMode, InteractionContext, InteractiveTool, PaneSidePanel};
use eframe::egui;

/// Computes a pane-salted panel [`egui::Id`] from the enclosing `ui` hierarchy.
#[inline]
pub fn scoped_panel_id(ui: &egui::Ui, suffix: impl std::hash::Hash) -> egui::Id {
    ui.id().with(suffix)
}

const FOCUSED_WIDGET_TYPE_KEY: &str = "FOCUSED_WIDGET_TYPE";

fn output_focused_widget_type(ctx: &egui::Context) -> Option<egui::WidgetType> {
    ctx.output(|o| {
        o.events.iter().rev().find_map(|ev| match ev {
            egui::output::OutputEvent::FocusGained(info)
            | egui::output::OutputEvent::Clicked(info) => Some(info.typ),
            _ => None,
        })
    })
}

/// Records the `egui::WidgetType` of the currently focused widget at the end of a frame pass.
pub fn record_focused_widget_type_from_output(ctx: &egui::Context) {
    let key = egui::Id::new(FOCUSED_WIDGET_TYPE_KEY);
    let Some(fid) = ctx.memory(|m| m.focused()) else {
        ctx.data_mut(|d| d.remove::<(egui::Id, egui::WidgetType)>(key));
        return;
    };
    if let Some(typ) = output_focused_widget_type(ctx) {
        ctx.data_mut(|d| d.insert_temp(key, (fid, typ)));
    } else {
        ctx.data_mut(|d| {
            if d.get_temp::<(egui::Id, egui::WidgetType)>(key)
                .is_some_and(|(cached_id, _)| cached_id != fid)
            {
                d.remove::<(egui::Id, egui::WidgetType)>(key);
            }
        });
    }
}

fn focused_widget_type(ctx: &egui::Context, fid: egui::Id) -> Option<egui::WidgetType> {
    if ctx.read_response(fid).is_some_and(|r| r.gained_focus()) {
        if let Some(typ) = output_focused_widget_type(ctx) {
            return Some(typ);
        }
    }
    ctx.data(|d| {
        d.get_temp::<(egui::Id, egui::WidgetType)>(egui::Id::new(FOCUSED_WIDGET_TYPE_KEY))
            .and_then(|(cached_id, typ)| (cached_id == fid).then_some(typ))
    })
}

/// Returns `true` when keyboard focus is currently held by a `WidgetType::TextEdit` input.
pub fn is_text_edit_focused(ctx: &egui::Context) -> bool {
    let Some(fid) = ctx.memory(|m| m.focused()) else {
        return false;
    };
    egui::text_edit::TextEditState::load(ctx, fid).is_some()
        || focused_widget_type(ctx, fid) == Some(egui::WidgetType::TextEdit)
}

/// Returns `true` when keyboard focus is held by a button or click-activated widget reserving `Space`.
pub fn is_button_activation_focused(ctx: &egui::Context) -> bool {
    let Some(fid) = ctx.memory(|m| m.focused()) else {
        return false;
    };
    if crate::app::viewport::sash::is_any_sash_id(fid) {
        return false;
    }
    if let Some(typ) = focused_widget_type(ctx, fid) {
        return matches!(
            typ,
            egui::WidgetType::Button
                | egui::WidgetType::Checkbox
                | egui::WidgetType::RadioButton
                | egui::WidgetType::SelectableLabel
                | egui::WidgetType::ComboBox
                | egui::WidgetType::CollapsingHeader
                | egui::WidgetType::Link
                | egui::WidgetType::ColorButton
        );
    }
    ctx.read_response(fid)
        .is_some_and(|resp| resp.sense.senses_click() && !resp.sense.senses_drag())
}

/// Returns `true` when unfocused pane-level keyboard shortcuts (e.g. `.`, `Ctrl+Z`) may execute in `ui`.
pub fn is_pane_shortcut_allowed(ui: &egui::Ui) -> bool {
    let ctx = ui.ctx();
    if is_text_edit_focused(ctx)
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

/// Returns `true` when activation-key shortcuts (`Space`) may execute without colliding with focused buttons.
pub fn is_pane_activation_shortcut_allowed(ui: &egui::Ui) -> bool {
    is_pane_shortcut_allowed(ui) && !is_button_activation_focused(ui.ctx())
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

fn render_default_tool_canvas<T: InteractiveTool + ?Sized>(tool: &mut T, ui: &mut egui::Ui) {
    egui::CentralPanel::default().show_inside(ui, |ui| {
        let (response, painter) =
            ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
        let multi_touch = ui.input(|i| i.multi_touch());
        let input_mode = ui.ctx().data(|d| {
            d.get_temp(egui::Id::new("INPUT_MODE"))
                .unwrap_or(InputMode::Mouse)
        });
        let keys_down = ui.input(|i| i.keys_down.clone());
        let modifiers = ui.input(|i| i.modifiers);

        let interaction_ctx = InteractionContext {
            pointer_pos: response.interact_pointer_pos(),
            delta: response.drag_delta(),
            is_dragging: response.dragged(),
            is_clicked: response.clicked(),
            response: &response,
            input_mode,
            multi_touch,
            keys_down,
            modifiers,
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
