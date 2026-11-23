use super::{SessionLifecycleState, WorkspaceSessionManager};
use crate::app::{MathExplorerApp, SplitLayoutMode, ViewportLayout};
use eframe::egui;
use egui_plot::commands::{CommandRegistryData, CommandTrigger};

#[cfg(feature = "pure_math")]
const WORKSPACE_HISTORY_BRIDGE_ID: &str = "WORKSPACE_HISTORY_BRIDGE";

#[cfg(feature = "pure_math")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PendingHistoryAction {
    Undo,
    Redo,
}

#[cfg(feature = "pure_math")]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WorkspaceHistoryBridge {
    pub(crate) active: bool,
    pub(crate) can_undo: bool,
    pub(crate) can_redo: bool,
    pub(crate) pending: Option<PendingHistoryAction>,
}

#[cfg(feature = "pure_math")]
pub(crate) fn read_workspace_history_bridge(ctx: &egui::Context) -> WorkspaceHistoryBridge {
    ctx.data(|d| {
        d.get_temp::<WorkspaceHistoryBridge>(egui::Id::new(WORKSPACE_HISTORY_BRIDGE_ID))
            .unwrap_or_default()
    })
}

#[cfg(feature = "pure_math")]
pub(crate) fn is_workspace_history_active(ctx: &egui::Context) -> bool {
    read_workspace_history_bridge(ctx).active
}

#[cfg(feature = "pure_math")]
pub(crate) fn request_workspace_undo(ctx: &egui::Context) {
    ctx.data_mut(|d| {
        let id = egui::Id::new(WORKSPACE_HISTORY_BRIDGE_ID);
        let mut bridge = d.get_temp::<WorkspaceHistoryBridge>(id).unwrap_or_default();
        bridge.pending = Some(PendingHistoryAction::Undo);
        d.insert_temp(id, bridge);
    });
}

#[cfg(feature = "pure_math")]
pub(crate) fn request_workspace_redo(ctx: &egui::Context) {
    ctx.data_mut(|d| {
        let id = egui::Id::new(WORKSPACE_HISTORY_BRIDGE_ID);
        let mut bridge = d.get_temp::<WorkspaceHistoryBridge>(id).unwrap_or_default();
        bridge.pending = Some(PendingHistoryAction::Redo);
        d.insert_temp(id, bridge);
    });
}

fn consume_and_strip_shortcut(
    ctx: &egui::Context,
    mods: egui::Modifiers,
    key: egui::Key,
    action_name: &str,
) -> bool {
    let triggered = ctx.input_mut(|i| {
        let consumed = i.consume_shortcut(&egui::KeyboardShortcut::new(mods, key));
        if consumed {
            i.keys_down.remove(&key);
            i.raw.events.retain(|ev| {
                !matches!(
                    ev,
                    egui::Event::Key {
                        key: ev_key,
                        modifiers: ev_mods,
                        pressed: true,
                        ..
                    } if *ev_key == key && ev_mods.matches_logically(mods)
                )
            });
        }
        consumed
    });
    if triggered {
        crate::accessibility::announce_live_status(ctx, format!("Action executed: {action_name}"));
    }
    triggered
}

impl MathExplorerApp {
    /// Returns true if an undo action is available on the workspace session.
    #[cfg(any(not(target_os = "macos"), test))]
    pub fn can_undo(&self) -> bool {
        self.session_manager.can_undo()
    }

    /// Returns true if a redo action is available on the workspace session.
    #[cfg(any(not(target_os = "macos"), test))]
    pub fn can_redo(&self) -> bool {
        self.session_manager.can_redo()
    }

    #[cfg(feature = "pure_math")]
    pub(crate) fn sync_workspace_history_bridge(&self, ctx: &egui::Context) {
        let bridge = WorkspaceHistoryBridge {
            active: true,
            can_undo: self.session_manager.can_undo(),
            can_redo: self.session_manager.can_redo(),
            pending: None,
        };
        ctx.data_mut(|d| d.insert_temp(egui::Id::new(WORKSPACE_HISTORY_BRIDGE_ID), bridge));
    }

    #[cfg(not(feature = "pure_math"))]
    pub(crate) fn sync_workspace_history_bridge(&self, _ctx: &egui::Context) {}

    #[cfg(feature = "pure_math")]
    pub(crate) fn drain_workspace_history_bridge(&mut self, ctx: &egui::Context) {
        let pending = ctx.data_mut(|d| {
            let id = egui::Id::new(WORKSPACE_HISTORY_BRIDGE_ID);
            let mut bridge = d.get_temp::<WorkspaceHistoryBridge>(id).unwrap_or_default();
            let action = bridge.pending.take();
            d.insert_temp(id, bridge);
            action
        });
        match pending {
            Some(PendingHistoryAction::Undo) => self.undo(),
            Some(PendingHistoryAction::Redo) => self.redo(),
            None => {}
        }
    }

    #[cfg(not(feature = "pure_math"))]
    pub(crate) fn drain_workspace_history_bridge(&mut self, _ctx: &egui::Context) {}

    pub(super) fn wants_non_canvas_keyboard_input(&self, ctx: &egui::Context) -> bool {
        if self.command_palette.is_open
            || self.show_help_menu
            || egui::Popup::is_any_open(ctx)
            || ctx.memory(|m| m.top_modal_layer().is_some())
        {
            return true;
        }
        if !ctx.wants_keyboard_input() {
            return false;
        }
        let Some(fid) = ctx.memory(|m| m.focused()) else {
            return true;
        };
        let pane_count = if self.viewport_layout != ViewportLayout::Single {
            self.viewport_layout.pane_count()
        } else if self.split_layout_mode != SplitLayoutMode::Single {
            2
        } else {
            1
        };
        !(0..pane_count)
            .any(|idx| self.pane_tab(idx).and_then(|t| t.focused_canvas_id()) == Some(fid))
    }
}

impl WorkspaceSessionManager {
    fn reg_shortcut(
        ctx: &egui::Context,
        name: &str,
        desc: &str,
        cat: &str,
        k: egui::Key,
        m: egui::Modifiers,
    ) -> bool {
        CommandRegistryData::register(ctx, name, desc, CommandTrigger::Shortcut(m, k), true, cat);
        let touch_mode = ctx.data(|d| {
            d.get_temp::<bool>(egui::Id::new("INPUT_MODE_TOUCH"))
                .unwrap_or(false)
        });
        !touch_mode && consume_and_strip_shortcut(ctx, m, k, name)
    }

    /// Registers global workspace session keyboard shortcuts and checks for user triggers.
    pub fn register_and_check_shortcuts(&mut self, ctx: &egui::Context, app: &mut MathExplorerApp) {
        let modifiers = if cfg!(target_os = "macos") {
            egui::Modifiers::MAC_CMD
        } else {
            egui::Modifiers::CTRL
        };
        let shift_modifiers = modifiers | egui::Modifiers::SHIFT;

        let (undo, redo) = Self::check_history_shortcuts(ctx, app, modifiers, shift_modifiers);
        let (open, save_as, save) = Self::check_file_shortcuts(ctx, modifiers, shift_modifiers);

        if undo {
            self.execute_undo(app);
        } else if redo {
            self.execute_redo(app);
        } else if open {
            self.trigger_open();
        } else if save_as {
            if self.lifecycle_state == SessionLifecycleState::DragEditing
                || crate::reflective_ui::slider::has_active_sweeps(ctx)
            {
                app.clear_active_pane_caches();
            }
            let state = app.get_app_state();
            let env_json = self.create_envelope(state);
            self.trigger_save_as(env_json);
        } else if save {
            if self.lifecycle_state == SessionLifecycleState::DragEditing
                || crate::reflective_ui::slider::has_active_sweeps(ctx)
            {
                app.clear_active_pane_caches();
            }
            let state = app.get_app_state();
            let env_json = self.create_envelope(state);
            self.trigger_save(env_json);
        }
    }

    fn register_history_metadata(ctx: &egui::Context, m: egui::Modifiers, sm: egui::Modifiers) {
        CommandRegistryData::register(
            ctx,
            "Undo",
            "Undo the last workspace state change",
            CommandTrigger::Shortcut(m, egui::Key::Z),
            true,
            "Edit",
        );
        CommandRegistryData::register(
            ctx,
            "Redo",
            "Redo the last undone workspace state change",
            CommandTrigger::Shortcut(sm, egui::Key::Z),
            true,
            "Edit",
        );
        if !cfg!(target_os = "macos") {
            CommandRegistryData::register(
                ctx,
                "Redo (Ctrl+Y)",
                "Redo the last undone workspace state change",
                CommandTrigger::Shortcut(m, egui::Key::Y),
                true,
                "Edit",
            );
        }
    }

    fn check_history_shortcuts(
        ctx: &egui::Context,
        app: &MathExplorerApp,
        m: egui::Modifiers,
        sm: egui::Modifiers,
    ) -> (bool, bool) {
        Self::register_history_metadata(ctx, m, sm);
        let touch_mode = ctx.data(|d| {
            d.get_temp::<bool>(egui::Id::new("INPUT_MODE_TOUCH"))
                .unwrap_or(false)
        });
        if touch_mode || app.wants_non_canvas_keyboard_input(ctx) {
            return (false, false);
        }
        let redo_shift = consume_and_strip_shortcut(ctx, sm, egui::Key::Z, "Redo");
        let redo_ctrl_y = !cfg!(target_os = "macos")
            && consume_and_strip_shortcut(ctx, m, egui::Key::Y, "Redo (Ctrl+Y)");
        let redo = redo_shift || redo_ctrl_y;
        let undo = !redo && consume_and_strip_shortcut(ctx, m, egui::Key::Z, "Undo");
        (undo, redo)
    }

    pub(crate) fn check_file_shortcuts(
        ctx: &egui::Context,
        m: egui::Modifiers,
        sm: egui::Modifiers,
    ) -> (bool, bool, bool) {
        let open = Self::reg_shortcut(
            ctx,
            "Open Workspace Session...",
            "Open a saved workspace session from disk",
            "File",
            egui::Key::O,
            m,
        );
        let save_as = Self::reg_shortcut(
            ctx,
            "Save Workspace Session As...",
            "Save the current workspace session to a new file",
            "File",
            egui::Key::S,
            sm,
        );
        let save = Self::reg_shortcut(
            ctx,
            "Save Workspace Session",
            "Save the current workspace session",
            "File",
            egui::Key::S,
            m,
        );
        (open, save_as, save)
    }
}
