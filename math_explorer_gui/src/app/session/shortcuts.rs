use super::WorkspaceSessionManager;
use crate::app::MathExplorerApp;
use eframe::egui;

impl WorkspaceSessionManager {
    fn reg_shortcut(
        ctx: &egui::Context,
        name: &str,
        desc: &str,
        cat: &str,
        k: egui::Key,
        m: egui::Modifiers,
    ) -> bool {
        egui_plot::commands::CommandRegistryData::register_and_check(
            ctx,
            name,
            desc,
            egui_plot::commands::CommandTrigger::Shortcut(m, k),
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

        let (undo, redo) = Self::check_history_shortcuts(ctx, modifiers, shift_modifiers);
        let (open, save_as, save) = Self::check_file_shortcuts(ctx, modifiers, shift_modifiers);

        if undo {
            app.undo();
        } else if redo {
            app.redo();
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

    fn check_history_shortcuts(
        ctx: &egui::Context,
        m: egui::Modifiers,
        sm: egui::Modifiers,
    ) -> (bool, bool) {
        let undo = Self::reg_shortcut(
            ctx,
            "Undo",
            "Undo the last workspace state change",
            "Edit",
            egui::Key::Z,
            m,
        );
        let redo_shift = Self::reg_shortcut(
            ctx,
            "Redo",
            "Redo the last undone workspace state change",
            "Edit",
            egui::Key::Z,
            sm,
        );
        let redo_ctrl_y = !cfg!(target_os = "macos")
            && Self::reg_shortcut(
                ctx,
                "Redo (Ctrl+Y)",
                "Redo the last undone workspace state change",
                "Edit",
                egui::Key::Y,
                m,
            );
        (undo, redo_shift || redo_ctrl_y)
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
