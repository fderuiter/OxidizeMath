#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::app::MathExplorerApp;
use eframe::egui;

/// Computes the `(can_undo, can_redo)` enabled state pair for the Edit menu.
pub(crate) fn edit_menu_enabled_states(app: &MathExplorerApp) -> (bool, bool) {
    (app.can_undo(), app.can_redo())
}

/// Dispatches `"edit_undo"` and `"edit_redo"` menu item actions and announces status.
pub(crate) fn handle_edit_menu_action(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    id: &str,
) -> bool {
    match id {
        "edit_undo" => {
            if app.can_undo() {
                app.undo();
                crate::accessibility::announce_live_status(ctx, "Action executed: Undo");
            }
            true
        }
        "edit_redo" => {
            if app.can_redo() {
                app.redo();
                crate::accessibility::announce_live_status(ctx, "Action executed: Redo");
            }
            true
        }
        _ => false,
    }
}

#[cfg(target_os = "macos")]
mod native {
    use super::edit_menu_enabled_states;
    use crate::app::MathExplorerApp;
    use muda::accelerator::{Accelerator, Code, Modifiers};
    use muda::{MenuId, MenuItem, Submenu};
    use std::cell::RefCell;

    struct EditMenuHandles {
        undo_item: MenuItem,
        redo_item: MenuItem,
        last_synced: (bool, bool),
    }

    thread_local! {
        static EDIT_MENU_HANDLES: RefCell<Option<EditMenuHandles>> = const { RefCell::new(None) };
    }

    pub(crate) fn undo_accelerator() -> Accelerator {
        Accelerator::new(Some(Modifiers::SUPER), Code::KeyZ)
    }

    pub(crate) fn redo_accelerator() -> Accelerator {
        Accelerator::new(Some(Modifiers::SUPER | Modifiers::SHIFT), Code::KeyZ)
    }

    pub(in crate::app::macos_menu) fn build_edit_menu(app: &MathExplorerApp) -> Submenu {
        let (can_undo, can_redo) = edit_menu_enabled_states(app);
        let edit_menu = Submenu::new("Edit", true);
        let undo_item = MenuItem::with_id(
            MenuId::new("edit_undo"),
            "Undo",
            can_undo,
            Some(undo_accelerator()),
        );
        let redo_item = MenuItem::with_id(
            MenuId::new("edit_redo"),
            "Redo",
            can_redo,
            Some(redo_accelerator()),
        );
        edit_menu.append(&undo_item).unwrap();
        edit_menu.append(&redo_item).unwrap();
        EDIT_MENU_HANDLES.with(|slot| {
            *slot.borrow_mut() = Some(EditMenuHandles {
                undo_item,
                redo_item,
                last_synced: (can_undo, can_redo),
            });
        });
        edit_menu
    }

    pub(crate) fn sync_macos_edit_menu(app: &MathExplorerApp) {
        if !super::super::can_init_nsapp_menu() {
            return;
        }
        let next = edit_menu_enabled_states(app);
        EDIT_MENU_HANDLES.with(|slot| {
            if let Some(handles) = slot.borrow_mut().as_mut() {
                if handles.last_synced.0 != next.0 {
                    handles.undo_item.set_enabled(next.0);
                }
                if handles.last_synced.1 != next.1 {
                    handles.redo_item.set_enabled(next.1);
                }
                handles.last_synced = next;
            }
        });
    }
}

#[cfg(target_os = "macos")]
pub(super) use native::build_edit_menu;
#[cfg(target_os = "macos")]
pub(crate) use native::sync_macos_edit_menu;
#[cfg(all(test, target_os = "macos"))]
pub(super) use native::{redo_accelerator, undo_accelerator};
