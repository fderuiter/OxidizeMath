use egui::{Context, Id, Key, Modifiers, Response, Ui};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommandTrigger {
    Key(Key),
    Shortcut(Modifiers, Key),
    AltClick,
}

impl CommandTrigger {
    /// Formats the trigger into a human-readable shortcut label.
    #[must_use]
    pub fn display_label(&self) -> String {
        match self {
            Self::Key(k) => format_key(*k).to_owned(),
            Self::Shortcut(m, k) => {
                let mut parts = Vec::with_capacity(4);
                if m.ctrl || (!cfg!(target_os = "macos") && m.command) {
                    parts.push("Ctrl");
                }
                if m.mac_cmd || (cfg!(target_os = "macos") && m.command) {
                    parts.push("Cmd");
                }
                if m.alt {
                    parts.push("Alt");
                }
                if m.shift {
                    parts.push("Shift");
                }
                parts.push(format_key(*k));
                parts.join("+")
            }
            Self::AltClick => "Alt+Click".to_owned(),
        }
    }
}

fn format_key(key: Key) -> &'static str {
    match key {
        Key::Questionmark => "?",
        Key::Slash => "/",
        Key::Plus => "+",
        Key::Equals => "=",
        Key::Minus => "-",
        Key::Home => "Home",
        Key::ArrowUp => "Up",
        Key::ArrowDown => "Down",
        Key::ArrowLeft => "Left",
        Key::ArrowRight => "Right",
        _ => key.name(),
    }
}

#[derive(Clone)]
pub struct CommandMetadata {
    pub name: String,
    pub description: String,
    pub trigger: CommandTrigger,
    pub desktop_only: bool,
    pub context: String,
}

#[derive(Default, Clone)]
pub struct CommandRegistryData {
    pub commands: Vec<CommandMetadata>,
}

impl CommandRegistryData {
    /// Records command metadata in `CMD_REGISTRY` without checking or consuming input events.
    pub fn register(
        ctx: &Context,
        name: &str,
        description: &str,
        trigger: CommandTrigger,
        desktop_only: bool,
        context_name: &str,
    ) {
        let mut data = ctx.data_mut(|d| {
            d.get_temp::<Self>(Id::new("CMD_REGISTRY"))
                .unwrap_or_default()
        });

        // Avoid duplicates in the same frame while allowing alternate triggers
        if !data
            .commands
            .iter()
            .any(|c| c.name == name && c.context == context_name && c.trigger == trigger)
        {
            data.commands.push(CommandMetadata {
                name: name.to_owned(),
                description: description.to_owned(),
                trigger,
                desktop_only,
                context: context_name.to_owned(),
            });
            ctx.data_mut(|d| d.insert_temp(Id::new("CMD_REGISTRY"), data));
        }
    }

    pub fn register_and_check(
        ctx: &Context,
        name: &str,
        description: &str,
        trigger: CommandTrigger,
        desktop_only: bool,
        context_name: &str,
        ui: Option<&Ui>,
        response: Option<&Response>,
    ) -> bool {
        Self::register(
            ctx,
            name,
            description,
            trigger.clone(),
            desktop_only,
            context_name,
        );

        let input_mode = ctx.data(|d| {
            d.get_temp::<bool>(Id::new("INPUT_MODE_TOUCH"))
                .unwrap_or(false)
        });
        if desktop_only && input_mode {
            return false;
        }

        let mut triggered = false;

        match trigger {
            CommandTrigger::Key(k) => {
                if let Some(ui) = ui {
                    if ui.input(|i| i.key_pressed(k)) {
                        triggered = true;
                    }
                } else if ctx.input(|i| i.key_pressed(k)) {
                    triggered = true;
                }
            }
            CommandTrigger::Shortcut(mods, k) => {
                if let Some(ui) = ui {
                    if ui.input_mut(|i| i.consume_shortcut(&egui::KeyboardShortcut::new(mods, k))) {
                        triggered = true;
                    }
                } else if ctx
                    .input_mut(|i| i.consume_shortcut(&egui::KeyboardShortcut::new(mods, k)))
                {
                    triggered = true;
                }
            }
            CommandTrigger::AltClick => {
                if let (Some(ui), Some(resp)) = (ui, response)
                    && resp.clicked()
                    && ui.input(|i| i.modifiers.alt)
                {
                    triggered = true;
                }
            }
        }

        if triggered {
            let msg = format!("Action executed: {name}");
            crate::emit_plot_live_message(ctx, msg, false);
        }

        triggered
    }

    /// Registers `Ctrl+Shift+I` (`Import CSV`) and `Ctrl+V` (`Paste CSV / TSV`) in `CMD_REGISTRY`.
    pub fn register_csv_import_commands(ctx: &Context) {
        Self::register(
            ctx,
            "Import CSV",
            "Import a CSV or TSV file as a 2D plot overlay",
            CommandTrigger::Shortcut(Modifiers::COMMAND | Modifiers::SHIFT, Key::I),
            false,
            "Plot Data",
        );
        Self::register(
            ctx,
            "Paste CSV / TSV",
            "Paste spreadsheet or delimited rows from clipboard onto focused plot",
            CommandTrigger::Shortcut(Modifiers::COMMAND, Key::V),
            false,
            "Plot Data",
        );
    }

    /// Checks and consumes `Ctrl+Shift+I` / `Cmd+Shift+I` to trigger CSV file import.
    pub fn check_import_csv_shortcut(ctx: &Context, ui: &Ui) -> bool {
        let cmd_shift = Self::register_and_check(
            ctx,
            "Import CSV",
            "Import a CSV or TSV file as a 2D plot overlay",
            CommandTrigger::Shortcut(Modifiers::COMMAND | Modifiers::SHIFT, Key::I),
            false,
            "Plot Data",
            Some(ui),
            None,
        );
        if cmd_shift {
            return true;
        }
        let ctrl_shift = egui::KeyboardShortcut::new(Modifiers::CTRL | Modifiers::SHIFT, Key::I);
        if ui.input_mut(|i| i.consume_shortcut(&ctrl_shift)) {
            crate::emit_plot_live_message(ctx, "Action executed: Import CSV".to_owned(), false);
            return true;
        }
        false
    }

    /// Registers `Alt+ArrowRight` and `Alt+ArrowLeft` critical point commands in `CMD_REGISTRY`.
    pub fn register_poi_navigation(ctx: &Context) {
        Self::register(
            ctx,
            "Next Critical Point",
            "Jump to the next critical point (root, extremum, or intersection)",
            CommandTrigger::Shortcut(Modifiers::ALT, Key::ArrowRight),
            false,
            "Plot Accessibility",
        );
        Self::register(
            ctx,
            "Previous Critical Point",
            "Jump to the previous critical point (root, extremum, or intersection)",
            CommandTrigger::Shortcut(Modifiers::ALT, Key::ArrowLeft),
            false,
            "Plot Accessibility",
        );
    }

    /// Checks `Alt+ArrowRight` (`Some(1)`) or `Alt+ArrowLeft` (`Some(-1)`) on a focused plot.
    pub fn check_poi_navigation_step(ctx: &Context, ui: &Ui) -> Option<isize> {
        if Self::register_and_check(
            ctx,
            "Next Critical Point",
            "Jump to the next critical point (root, extremum, or intersection)",
            CommandTrigger::Shortcut(Modifiers::ALT, Key::ArrowRight),
            false,
            "Plot Accessibility",
            Some(ui),
            None,
        ) {
            return Some(1);
        }
        if Self::register_and_check(
            ctx,
            "Previous Critical Point",
            "Jump to the previous critical point (root, extremum, or intersection)",
            CommandTrigger::Shortcut(Modifiers::ALT, Key::ArrowLeft),
            false,
            "Plot Accessibility",
            Some(ui),
            None,
        ) {
            return Some(-1);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_records_metadata_without_consuming_events() {
        let ctx = Context::default();
        let raw_input = egui::RawInput {
            events: vec![egui::Event::Key {
                key: Key::K,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Modifiers::CTRL,
            }],
            modifiers: Modifiers::CTRL,
            ..Default::default()
        };

        let _ = ctx.run(raw_input, |ctx| {
            CommandRegistryData::register(
                ctx,
                "Command Palette",
                "Open command palette",
                CommandTrigger::Shortcut(Modifiers::CTRL, Key::K),
                true,
                "Global",
            );
            // Key event must remain unconsumed after register()
            let still_present = ctx.input_mut(|i| {
                i.consume_shortcut(&egui::KeyboardShortcut::new(Modifiers::CTRL, Key::K))
            });
            assert!(still_present);
        });

        let reg = ctx.data(|d| {
            d.get_temp::<CommandRegistryData>(Id::new("CMD_REGISTRY"))
                .unwrap_or_default()
        });
        assert_eq!(reg.commands.len(), 1);
        assert_eq!(reg.commands[0].name, "Command Palette");
        assert!(
            ctx.data(|d| d.get_temp::<String>(Id::new("aria_live_message")))
                .is_none()
        );
    }

    #[test]
    fn test_register_deduplicates_identical_trigger_and_allows_alternate_trigger() {
        let ctx = Context::default();
        CommandRegistryData::register(
            &ctx,
            "Keyboard Shortcuts",
            "Toggle keyboard shortcuts modal",
            CommandTrigger::Shortcut(Modifiers::CTRL, Key::Slash),
            true,
            "Global",
        );
        CommandRegistryData::register(
            &ctx,
            "Keyboard Shortcuts",
            "Toggle keyboard shortcuts modal",
            CommandTrigger::Shortcut(Modifiers::CTRL, Key::Slash),
            true,
            "Global",
        );
        CommandRegistryData::register(
            &ctx,
            "Keyboard Shortcuts",
            "Toggle keyboard shortcuts modal",
            CommandTrigger::Key(Key::Questionmark),
            true,
            "Global",
        );

        let reg = ctx.data(|d| {
            d.get_temp::<CommandRegistryData>(Id::new("CMD_REGISTRY"))
                .unwrap_or_default()
        });
        assert_eq!(reg.commands.len(), 2);
    }

    #[test]
    fn test_display_label_formats_readable_keys() {
        assert_eq!(CommandTrigger::Key(Key::Questionmark).display_label(), "?");
        assert_eq!(CommandTrigger::Key(Key::Plus).display_label(), "+");
        assert_eq!(CommandTrigger::Key(Key::Equals).display_label(), "=");
        assert_eq!(CommandTrigger::Key(Key::Minus).display_label(), "-");
        assert_eq!(CommandTrigger::Key(Key::Home).display_label(), "Home");
        assert_eq!(CommandTrigger::Key(Key::ArrowDown).display_label(), "Down");
        assert_eq!(
            CommandTrigger::Shortcut(Modifiers::CTRL | Modifiers::SHIFT, Key::D).display_label(),
            "Ctrl+Shift+D"
        );
        assert_eq!(
            CommandTrigger::Shortcut(Modifiers::MAC_CMD, Key::Slash).display_label(),
            "Cmd+/"
        );
        assert_eq!(
            CommandTrigger::Shortcut(Modifiers::ALT, Key::ArrowRight).display_label(),
            "Alt+Right"
        );
        assert_eq!(
            CommandTrigger::Shortcut(Modifiers::ALT, Key::ArrowLeft).display_label(),
            "Alt+Left"
        );
        assert_eq!(CommandTrigger::AltClick.display_label(), "Alt+Click");
    }
}
