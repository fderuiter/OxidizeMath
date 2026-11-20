use super::WorkspaceSessionManager;
use crate::app::{MathExplorerApp, SplitLayoutMode, ViewportLayout};
use eframe::egui;
use egui_plot::commands::{CommandRegistryData, CommandTrigger};

fn consume_and_strip_shortcut(
    ctx: &egui::Context,
    mods: egui::Modifiers,
    key: egui::Key,
    action_name: &str,
) -> bool {
    let triggered = ctx.input_mut(|i| {
        let consumed = i.consume_shortcut(&egui::KeyboardShortcut::new(mods, key));
        if consumed {
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
    /// Returns true if an undo action is available on the focused tool or workspace session.
    pub fn can_undo(&self) -> bool {
        let target = self.active_focused_pane();
        self.pane_tab(target).is_some_and(|t| t.can_undo()) || self.session_manager.can_undo()
    }

    /// Returns true if a redo action is available on the focused tool or workspace session.
    pub fn can_redo(&self) -> bool {
        let target = self.active_focused_pane();
        self.pane_tab(target).is_some_and(|t| t.can_redo()) || self.session_manager.can_redo()
    }

    pub(crate) fn try_focused_pane_undo(&mut self) -> bool {
        if !self
            .pane_tab(self.active_focused_pane())
            .is_some_and(|t| t.can_undo())
        {
            return false;
        }
        let Some((handled, tab_name)) = self
            .focused_pane_tab_mut()
            .map(|t| (t.try_undo(), t.name().to_string()))
        else {
            return false;
        };
        if !handled {
            return false;
        }
        self.tab_state_cache.invalidate_tab(&tab_name);
        self.snapshot_focused_secondary_slot();
        let mut sm = std::mem::take(&mut self.session_manager);
        sm.sync_after_local_undo(self);
        self.session_manager = sm;
        true
    }

    pub(crate) fn try_focused_pane_redo(&mut self) -> bool {
        if !self
            .pane_tab(self.active_focused_pane())
            .is_some_and(|t| t.can_redo())
        {
            return false;
        }
        let prev = self.session_manager.last_committed_state.clone();
        let Some((handled, tab_name)) = self
            .focused_pane_tab_mut()
            .map(|t| (t.try_redo(), t.name().to_string()))
        else {
            return false;
        };
        if !handled {
            return false;
        }
        self.tab_state_cache.invalidate_tab(&tab_name);
        self.snapshot_focused_secondary_slot();
        let mut sm = std::mem::take(&mut self.session_manager);
        sm.sync_after_local_redo(prev, self);
        self.session_manager = sm;
        true
    }

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
        CommandRegistryData::register_and_check(
            ctx,
            name,
            desc,
            CommandTrigger::Shortcut(m, k),
            true,
            cat,
            None,
            None,
        )
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
            std::mem::swap(self, &mut app.session_manager);
            app.undo();
            std::mem::swap(self, &mut app.session_manager);
        } else if redo {
            std::mem::swap(self, &mut app.session_manager);
            app.redo();
            std::mem::swap(self, &mut app.session_manager);
        } else if open {
            self.trigger_open();
        } else if save_as {
            let state = app.get_app_state();
            let env_json = self.create_envelope(state);
            self.trigger_save_as(env_json);
        } else if save {
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

    fn check_file_shortcuts(
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
