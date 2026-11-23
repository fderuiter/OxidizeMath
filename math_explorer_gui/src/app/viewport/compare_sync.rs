#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::app::MathExplorerApp;
use crate::framework::PlaybackAction;
use eframe::egui;
use serde::{Deserialize, Serialize};

const SYNC_PLAYBACK_STATE_KEY: &str = "COMPARE_SYNC_PLAYBACK_STATE";

/// Persistent multi-pane comparison synchronization toggles stored in `AppState`.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ComparisonSyncConfig {
    /// Synchronizes X and Y coordinate bounds (including union auto-bounds) across matching 2D plots.
    #[serde(default)]
    pub link_axes: bool,
    /// Synchronizes hover crosshair cursors across matching 2D plots.
    #[serde(default)]
    pub link_cursors: bool,
    /// Broadcasts `Play`, `Pause`, `Step`, and `Reset` across all active `PlaybackTransport` panes.
    #[serde(default)]
    pub sync_playback: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PlaybackSyncFrameState {
    pub enabled: bool,
    pub frame_seq: u64,
    pub pending_key: Option<PendingSyncKey>,
    pub active_broadcast: Option<PlaybackBroadcastEvent>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PendingSyncKey {
    TogglePlayPause,
    StepForward,
}

#[derive(Clone, Debug)]
pub(crate) struct PlaybackBroadcastEvent {
    pub action: PlaybackAction,
    pub emitted_frame: u64,
    pub announced: bool,
    pub consumed_key: Option<egui::Key>,
    pub delivered_ids: Vec<egui::Id>,
}

fn detect_workspace_sync_key(ctx: &egui::Context) -> Option<PendingSyncKey> {
    let shell_or_modal = crate::framework::is_text_edit_focused(ctx)
        || egui::Popup::is_any_open(ctx)
        || ctx.memory(|m| m.top_modal_layer().is_some())
        || ctx.data(|d| {
            d.get_temp::<bool>(egui::Id::new(super::focus::SHELL_MODAL_OR_SASH_FOCUSED_KEY))
                .unwrap_or(false)
        });
    if shell_or_modal {
        return None;
    }
    let space_allowed = !crate::framework::is_button_activation_focused(ctx);
    ctx.input(|i| {
        if space_allowed && i.key_pressed(egui::Key::Space) && i.modifiers.is_none() {
            Some(PendingSyncKey::TogglePlayPause)
        } else if i.key_pressed(egui::Key::Period) && i.modifiers.is_none() {
            Some(PendingSyncKey::StepForward)
        } else {
            None
        }
    })
}

/// Publishes multi-pane plot and playback comparison sync state into `egui::Context` for the frame.
pub(crate) fn publish_multi_pane_sync_context(app: &MathExplorerApp, ctx: &egui::Context) {
    let cfg = app.compare_sync;
    egui_plot::set_ambient_plot_link_config(
        ctx,
        egui_plot::AmbientPlotLinkConfig {
            link_axes: cfg.link_axes,
            link_cursors: cfg.link_cursors,
        },
    );
    let pending_key = cfg
        .sync_playback
        .then(|| detect_workspace_sync_key(ctx))
        .flatten();
    ctx.data_mut(|d| {
        let state: &mut PlaybackSyncFrameState =
            d.get_temp_mut_or_default(egui::Id::new(SYNC_PLAYBACK_STATE_KEY));
        state.frame_seq = state.frame_seq.wrapping_add(1);
        state.enabled = cfg.sync_playback;
        state.pending_key = pending_key;
        if !cfg.sync_playback {
            state.active_broadcast = None;
        } else if let Some(ref bcast) = state.active_broadcast {
            if state.frame_seq.saturating_sub(bcast.emitted_frame) > 1 {
                state.active_broadcast = None;
            }
        }
    });
}

/// Clears multi-pane comparison sync flags on `egui::Context` when rendering single-pane mode.
pub(super) fn clear_single_pane_sync_context(ctx: &egui::Context) {
    egui_plot::set_ambient_plot_link_config(ctx, egui_plot::AmbientPlotLinkConfig::default());
    ctx.data_mut(|d| {
        let state: &mut PlaybackSyncFrameState =
            d.get_temp_mut_or_default(egui::Id::new(SYNC_PLAYBACK_STATE_KEY));
        state.frame_seq = state.frame_seq.wrapping_add(1);
        state.enabled = false;
        state.pending_key = None;
        state.active_broadcast = None;
    });
}

/// Consumes a pre-detected workspace sync shortcut (`Space` or `Period`) for the first eligible transport.
pub(crate) fn claim_pending_sync_key(
    ctx: &egui::Context,
    can_toggle: bool,
    is_playing: bool,
    can_step: bool,
) -> Option<(PlaybackAction, egui::Key)> {
    let (key, action) = ctx.data_mut(|d| {
        let state: &mut PlaybackSyncFrameState =
            d.get_temp_mut_or_default(egui::Id::new(SYNC_PLAYBACK_STATE_KEY));
        if !state.enabled {
            return None;
        }
        match state.pending_key {
            Some(PendingSyncKey::TogglePlayPause) if can_toggle => {
                state.pending_key = None;
                Some((
                    egui::Key::Space,
                    if is_playing {
                        PlaybackAction::Pause
                    } else {
                        PlaybackAction::Play
                    },
                ))
            }
            Some(PendingSyncKey::StepForward) if can_step => {
                state.pending_key = None;
                Some((egui::Key::Period, PlaybackAction::Step))
            }
            _ => None,
        }
    })?;
    ctx.input_mut(|i| {
        let _ = i.consume_key(egui::Modifiers::NONE, key);
    });
    Some((action, key))
}

/// Emits a new synchronized playback broadcast from `origin_id` and returns whether this caller should announce.
pub(crate) fn emit_playback_broadcast(
    ctx: &egui::Context,
    origin_id: egui::Id,
    action: PlaybackAction,
    consumed_key: Option<egui::Key>,
) -> bool {
    ctx.data_mut(|d| {
        let state: &mut PlaybackSyncFrameState =
            d.get_temp_mut_or_default(egui::Id::new(SYNC_PLAYBACK_STATE_KEY));
        if !state.enabled {
            return true;
        }
        state.pending_key = None;
        state.active_broadcast = Some(PlaybackBroadcastEvent {
            action,
            emitted_frame: state.frame_seq,
            announced: true,
            consumed_key,
            delivered_ids: vec![origin_id],
        });
        true
    })
}

/// Polls whether an active synchronized playback broadcast applies to `transport_id`.
/// Returns `(action, should_announce)` when delivered.
pub(crate) fn poll_playback_broadcast(
    ctx: &egui::Context,
    transport_id: egui::Id,
) -> Option<(PlaybackAction, bool)> {
    let (action, should_announce, maybe_key) = ctx.data_mut(|d| {
        let state: &mut PlaybackSyncFrameState =
            d.get_temp_mut_or_default(egui::Id::new(SYNC_PLAYBACK_STATE_KEY));
        if !state.enabled {
            return None;
        }
        let seq = state.frame_seq;
        let bcast = state.active_broadcast.as_mut()?;
        if seq.saturating_sub(bcast.emitted_frame) > 1
            || bcast.delivered_ids.contains(&transport_id)
        {
            return None;
        }
        bcast.delivered_ids.push(transport_id);
        let should_announce = !bcast.announced;
        bcast.announced = true;
        Some((bcast.action, should_announce, bcast.consumed_key))
    })?;
    if let Some(key) = maybe_key {
        ctx.input_mut(|i| {
            let _ = i.consume_key(egui::Modifiers::NONE, key);
        });
    }
    Some((action, should_announce))
}

/// Renders comparison sync checkboxes (`Link Plot Axes`, `Link Plot Cursors`, `Sync Playback`).
pub(crate) fn render_comparison_sync_checkboxes(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    ui: &mut egui::Ui,
) {
    let mut link_axes = app.compare_sync.link_axes;
    if ui.checkbox(&mut link_axes, "Link Plot Axes").changed() {
        app.set_link_plot_axes(Some(ctx), link_axes);
    }
    let mut link_cursors = app.compare_sync.link_cursors;
    if ui
        .checkbox(&mut link_cursors, "Link Plot Cursors")
        .changed()
    {
        app.set_link_plot_cursors(Some(ctx), link_cursors);
    }
    let mut sync_playback = app.compare_sync.sync_playback;
    if ui.checkbox(&mut sync_playback, "Sync Playback").changed() {
        app.set_sync_playback(Some(ctx), sync_playback);
    }
}

impl MathExplorerApp {
    /// Updates `Link Plot Axes` and emits a polite ARIA live announcement when `ctx` is provided.
    pub fn set_link_plot_axes(&mut self, ctx: Option<&egui::Context>, enabled: bool) {
        if self.compare_sync.link_axes == enabled {
            return;
        }
        self.compare_sync.link_axes = enabled;
        if let Some(c) = ctx {
            let msg = if enabled {
                "Plot axis linking enabled across comparison panes"
            } else {
                "Plot axis linking disabled"
            };
            crate::accessibility::announce_live_status(c, msg);
            c.request_repaint();
        }
    }

    /// Updates `Link Plot Cursors` and emits a polite ARIA live announcement when `ctx` is provided.
    pub fn set_link_plot_cursors(&mut self, ctx: Option<&egui::Context>, enabled: bool) {
        if self.compare_sync.link_cursors == enabled {
            return;
        }
        self.compare_sync.link_cursors = enabled;
        if let Some(c) = ctx {
            let msg = if enabled {
                "Plot cursor linking enabled across comparison panes"
            } else {
                "Plot cursor linking disabled"
            };
            crate::accessibility::announce_live_status(c, msg);
            c.request_repaint();
        }
    }

    /// Updates `Sync Playback` and emits a polite ARIA live announcement when `ctx` is provided.
    pub fn set_sync_playback(&mut self, ctx: Option<&egui::Context>, enabled: bool) {
        if self.compare_sync.sync_playback == enabled {
            return;
        }
        self.compare_sync.sync_playback = enabled;
        if let Some(c) = ctx {
            let msg = if enabled {
                "Synchronized simulation playback enabled across panes"
            } else {
                "Synchronized simulation playback disabled"
            };
            crate::accessibility::announce_live_status(c, msg);
            c.request_repaint();
        }
    }
}
