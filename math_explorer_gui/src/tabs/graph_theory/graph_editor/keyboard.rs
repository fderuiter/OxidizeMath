#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GraphEditorTool;
use crate::accessibility::announce_status;
use crate::framework::InteractionContext;
use eframe::egui::{self, Pos2};
use egui_plot::commands::{CommandRegistryData, CommandTrigger};

/// Default translation step (in canvas points) for unmodified arrow key presses.
pub const TRANSLATE_STEP_NORMAL: f32 = 10.0;

/// Coarse translation step (in canvas points) when `Shift` is held with an arrow key.
pub const TRANSLATE_STEP_SHIFT: f32 = 50.0;

const NODE_OVERLAP_DISTANCE: f32 = 15.0;
const NODE_OFFSET_STEP: f32 = 30.0;
const MAX_OFFSET_ATTEMPTS: usize = 32;
const FALLBACK_VIEWPORT_CENTER: Pos2 = Pos2::new(400.0, 300.0);

fn is_key_triggered(ctx: &InteractionContext, key: egui::Key) -> bool {
    if ctx.response.ctx.input(|i| i.key_pressed(key)) {
        return true;
    }
    ctx.keys_down.contains(&key) && !ctx.response.ctx.input(|i| i.keys_down.contains(&key))
}

impl GraphEditorTool {
    pub(super) fn register_keyboard_commands(&self, ctx: &egui::Context) -> egui::Modifiers {
        let context = "Graph Editor";
        CommandRegistryData::register(
            ctx,
            "Select Next Node",
            "Cycle selection forward to the next graph node",
            CommandTrigger::Key(egui::Key::Tab),
            false,
            context,
        );
        CommandRegistryData::register(
            ctx,
            "Select Next Node",
            "Cycle selection forward to the next graph node",
            CommandTrigger::Key(egui::Key::OpenBracket),
            false,
            context,
        );
        CommandRegistryData::register(
            ctx,
            "Select Previous Node",
            "Cycle selection backward to the previous graph node",
            CommandTrigger::Shortcut(egui::Modifiers::SHIFT, egui::Key::Tab),
            false,
            context,
        );
        CommandRegistryData::register(
            ctx,
            "Select Previous Node",
            "Cycle selection backward to the previous graph node",
            CommandTrigger::Key(egui::Key::CloseBracket),
            false,
            context,
        );
        CommandRegistryData::register(
            ctx,
            "Add Node",
            "Insert a new node at the viewport center or non-overlapping offset",
            CommandTrigger::Key(egui::Key::N),
            false,
            context,
        );
        CommandRegistryData::register(
            ctx,
            "Connect Edge",
            "Connect the selected node to the next targeted node",
            CommandTrigger::Key(egui::Key::E),
            false,
            context,
        );
        self.register_arrow_commands(ctx);
        self.register_history_commands(ctx)
    }

    fn register_history_commands(&self, ctx: &egui::Context) -> egui::Modifiers {
        let m = if cfg!(target_os = "macos") {
            egui::Modifiers::MAC_CMD
        } else {
            egui::Modifiers::CTRL
        };
        CommandRegistryData::register(
            ctx,
            "Undo",
            "Undo the last graph modification",
            CommandTrigger::Shortcut(m, egui::Key::Z),
            true,
            "Graph Editor",
        );
        CommandRegistryData::register(
            ctx,
            "Redo",
            "Redo the last undone graph modification",
            CommandTrigger::Shortcut(m | egui::Modifiers::SHIFT, egui::Key::Z),
            true,
            "Graph Editor",
        );
        if !cfg!(target_os = "macos") {
            CommandRegistryData::register(
                ctx,
                "Redo",
                "Redo the last undone graph modification",
                CommandTrigger::Shortcut(m, egui::Key::Y),
                true,
                "Graph Editor",
            );
        }
        m
    }

    fn register_arrow_commands(&self, ctx: &egui::Context) {
        let arrows = [
            (egui::Key::ArrowLeft, "Move Node Left"),
            (egui::Key::ArrowRight, "Move Node Right"),
            (egui::Key::ArrowUp, "Move Node Up"),
            (egui::Key::ArrowDown, "Move Node Down"),
        ];
        for (key, name) in arrows {
            CommandRegistryData::register(
                ctx,
                name,
                "Translate the selected node by 10 points (50 points with Shift)",
                CommandTrigger::Key(key),
                false,
                "Graph Editor",
            );
        }
    }

    pub(super) fn configure_canvas_focus(&mut self, ui: &mut egui::Ui, response: &egui::Response) {
        self.canvas_id = Some(response.id);
        if response.clicked() || response.drag_started() {
            response.request_focus();
        }
        if response.has_focus() {
            ui.memory_mut(|m| {
                m.set_focus_lock_filter(
                    response.id,
                    egui::EventFilter {
                        tab: !self.node_positions.is_empty(),
                        horizontal_arrows: true,
                        vertical_arrows: true,
                        escape: false,
                    },
                );
            });
        }
    }

    pub(super) fn consume_toolbar_undo_redo(&mut self, ui: &mut egui::Ui, m: egui::Modifiers) {
        if !crate::framework::is_pane_shortcut_allowed(ui) {
            return;
        }
        let sm = m | egui::Modifiers::SHIFT;
        let (undo, redo) = ui.input_mut(|i| {
            let redo_shift = i.consume_shortcut(&egui::KeyboardShortcut::new(sm, egui::Key::Z));
            let redo_y = !cfg!(target_os = "macos")
                && i.consume_shortcut(&egui::KeyboardShortcut::new(m, egui::Key::Y));
            let redo = redo_shift || redo_y;
            let undo = !redo && i.consume_shortcut(&egui::KeyboardShortcut::new(m, egui::Key::Z));
            (undo, redo)
        });
        if redo && self.can_redo() {
            self.redo();
            announce_status("Redid graph modification");
        } else if undo && self.can_undo() {
            self.undo();
            announce_status("Undid graph modification");
        }
    }

    pub(super) fn handle_keyboard(&mut self, ctx: &InteractionContext) {
        let ctrl_or_cmd = ctx.modifiers.command || ctx.modifiers.ctrl || ctx.modifiers.mac_cmd;
        if ctrl_or_cmd {
            if !crate::app::session::shortcuts::is_workspace_history_active(&ctx.response.ctx) {
                self.handle_undo_redo_keys(ctx);
            }
            return;
        }
        if ctx.modifiers.alt {
            return;
        }
        self.handle_selection_keys(ctx);
        self.handle_creation_and_edge_keys(ctx);
        self.handle_translation_keys(ctx);
    }

    fn handle_undo_redo_keys(&mut self, ctx: &InteractionContext) {
        if is_key_triggered(ctx, egui::Key::Z) {
            if ctx.modifiers.shift {
                if self.can_redo() {
                    self.redo();
                    announce_status("Redid graph modification");
                }
            } else if self.can_undo() {
                self.undo();
                announce_status("Undid graph modification");
            }
        } else if is_key_triggered(ctx, egui::Key::Y) && self.can_redo() {
            self.redo();
            announce_status("Redid graph modification");
        }
    }

    fn sorted_node_ids(&self) -> Vec<usize> {
        let mut ids = Vec::with_capacity(self.node_positions.len());
        ids.extend(self.node_positions.keys().copied());
        ids.sort_unstable();
        ids
    }

    fn handle_selection_keys(&mut self, ctx: &InteractionContext) {
        let tab_pressed = is_key_triggered(ctx, egui::Key::Tab);
        let forward =
            (tab_pressed && !ctx.modifiers.shift) || is_key_triggered(ctx, egui::Key::OpenBracket);
        let backward =
            (tab_pressed && ctx.modifiers.shift) || is_key_triggered(ctx, egui::Key::CloseBracket);

        if forward {
            self.cycle_node_selection(true);
        } else if backward {
            self.cycle_node_selection(false);
        }
    }

    pub(crate) fn cycle_node_selection(&mut self, forward: bool) {
        let sorted_ids = self.sorted_node_ids();
        if sorted_ids.is_empty() {
            self.selected_node = None;
            return;
        }
        let next_id = match self.selected_node {
            Some(current) => match sorted_ids.binary_search(&current) {
                Ok(idx) if forward => sorted_ids[(idx + 1) % sorted_ids.len()],
                Ok(idx) => {
                    let prev = if idx == 0 {
                        sorted_ids.len() - 1
                    } else {
                        idx - 1
                    };
                    sorted_ids[prev]
                }
                Err(insert_idx) if forward => sorted_ids[insert_idx % sorted_ids.len()],
                Err(_) => sorted_ids[sorted_ids.len() - 1],
            },
            None if forward => sorted_ids[0],
            None => sorted_ids[sorted_ids.len() - 1],
        };
        if self.selected_node != Some(next_id) {
            self.selected_node = Some(next_id);
            announce_status(&format!("Selected node {next_id}"));
        }
    }

    fn handle_creation_and_edge_keys(&mut self, ctx: &InteractionContext) {
        if is_key_triggered(ctx, egui::Key::N) || is_key_triggered(ctx, egui::Key::Enter) {
            self.insert_node_from_keyboard(ctx.response.rect);
        } else if is_key_triggered(ctx, egui::Key::E) {
            self.connect_selected_node_to_next();
        }
    }

    fn resolve_insertion_position(&self, viewport: egui::Rect) -> Pos2 {
        let base = if viewport.is_positive()
            && viewport.center().x.is_finite()
            && viewport.center().y.is_finite()
        {
            viewport.center()
        } else {
            FALLBACK_VIEWPORT_CENTER
        };
        for step in 0..MAX_OFFSET_ATTEMPTS {
            let offset = step as f32 * NODE_OFFSET_STEP;
            let candidate = base + egui::vec2(offset, offset);
            let overlaps = self
                .node_positions
                .values()
                .any(|&existing| (existing - candidate).length() < NODE_OVERLAP_DISTANCE);
            if !overlaps {
                return candidate;
            }
        }
        base + egui::vec2(
            (self.next_node_id % MAX_OFFSET_ATTEMPTS) as f32 * NODE_OFFSET_STEP,
            NODE_OFFSET_STEP,
        )
    }

    pub(crate) fn insert_node_from_keyboard(&mut self, viewport: egui::Rect) {
        let pos = self.resolve_insertion_position(viewport);
        self.push_undo_snapshot();
        let node_id = self.next_node_id;
        let node_index = self.graph.add_node(pos);
        self.node_indices.insert(node_id, node_index);
        self.node_positions.insert(node_id, pos);
        self.next_node_id += 1;
        self.selected_node = Some(node_id);
        announce_status(&format!(
            "Added node {node_id} at ({:.1}, {:.1})",
            pos.x, pos.y
        ));
    }

    fn handle_translation_keys(&mut self, ctx: &InteractionContext) {
        let step = if ctx.modifiers.shift {
            TRANSLATE_STEP_SHIFT
        } else {
            TRANSLATE_STEP_NORMAL
        };
        let mut delta = egui::Vec2::ZERO;
        if is_key_triggered(ctx, egui::Key::ArrowLeft) {
            delta.x -= step;
        }
        if is_key_triggered(ctx, egui::Key::ArrowRight) {
            delta.x += step;
        }
        if is_key_triggered(ctx, egui::Key::ArrowUp) {
            delta.y -= step;
        }
        if is_key_triggered(ctx, egui::Key::ArrowDown) {
            delta.y += step;
        }
        if delta != egui::Vec2::ZERO {
            self.translate_selected_node(delta);
        }
    }

    pub(crate) fn translate_selected_node(&mut self, delta: egui::Vec2) {
        let Some(selected_id) = self.selected_node else {
            return;
        };
        if !self.node_positions.contains_key(&selected_id) {
            return;
        }
        self.push_undo_snapshot();
        let Some(pos) = self.node_positions.get_mut(&selected_id) else {
            return;
        };
        *pos += delta;
        let updated_pos = *pos;
        if let Some(&idx) = self.node_indices.get(&selected_id) {
            if let Some(weight) = self.graph.graph.node_weight_mut(idx) {
                *weight = updated_pos;
            }
        }
        announce_status(&format!(
            "Moved node {selected_id} to ({:.1}, {:.1})",
            updated_pos.x, updated_pos.y
        ));
    }

    pub(crate) fn connect_selected_node_to_next(&mut self) {
        let sorted_ids = self.sorted_node_ids();
        let Some(selected_id) = self
            .selected_node
            .filter(|id| self.node_positions.contains_key(id))
        else {
            announce_status("No node selected to connect");
            return;
        };
        let Ok(idx) = sorted_ids.binary_search(&selected_id) else {
            announce_status("No node selected to connect");
            return;
        };
        let target_id = sorted_ids[(idx + 1) % sorted_ids.len()];
        if target_id == selected_id {
            announce_status(&format!("Cannot create self-loop on node {selected_id}"));
            return;
        }
        let edge_exists = self.edges.iter().any(|&(u, v, _)| {
            (u == selected_id && v == target_id) || (v == selected_id && u == target_id)
        });
        if edge_exists {
            announce_status(&format!(
                "Duplicate edge between node {selected_id} and node {target_id} already exists"
            ));
            return;
        }
        self.push_undo_snapshot();
        let weight = 1.0;
        self.edges.push((selected_id, target_id, weight));
        if let (Some(&u_idx), Some(&v_idx)) = (
            self.node_indices.get(&selected_id),
            self.node_indices.get(&target_id),
        ) {
            self.graph.add_edge(u_idx, v_idx, weight);
        } else {
            self.rebuild_graph();
        }
        announce_status(&format!("Connected node {selected_id} to node {target_id}"));
    }
}
