//! Accessible sonification mute/unmute and gain controls rendered inside [`super::PlaybackTransport`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::playback::{scoped_audio_bridge_id, ScopedAudioUiBridge};
use crate::accessibility::{announce_status, AccessibleHoverText};
use crate::audio::AcousticSettings;
use eframe::egui;

pub(super) fn render_audio_controls(
    ui: &mut egui::Ui,
    base_id: egui::Id,
    tool_name: &str,
    ui_enabled: bool,
    min_size: egui::Vec2,
    settings: &mut AcousticSettings,
) {
    let bridge_id = scoped_audio_bridge_id();
    let standalone_id = base_id.with("standalone_muted");
    let (has_bridge, muted) = ui.ctx().data(|d| {
        if let Some(bridge) = d.get_temp::<ScopedAudioUiBridge>(bridge_id) {
            (true, bridge.muted)
        } else {
            let m = d.get_temp::<bool>(standalone_id).unwrap_or(true);
            (false, m)
        }
    });

    if render_mute_toggle(ui, base_id, tool_name, ui_enabled, min_size, muted) {
        let next_muted = !muted;
        if has_bridge {
            ui.ctx().data_mut(|d| {
                if let Some(mut b) = d.get_temp::<ScopedAudioUiBridge>(bridge_id) {
                    b.muted = next_muted;
                    b.toggled_mute = true;
                    d.insert_temp(bridge_id, b);
                }
            });
        } else {
            let msg = if next_muted {
                format!("Audio muted for {tool_name}")
            } else {
                format!("Audio unmuted for {tool_name}")
            };
            announce_status(&msg);
            ui.ctx()
                .data_mut(|d| d.insert_temp(standalone_id, next_muted));
        }
        ui.ctx().request_repaint();
    }

    if render_gain_slider(ui, base_id, tool_name, ui_enabled, settings) {
        *settings = settings.sanitized();
        if has_bridge {
            let clean = *settings;
            ui.ctx().data_mut(|d| {
                if let Some(mut b) = d.get_temp::<ScopedAudioUiBridge>(bridge_id) {
                    b.settings = clean;
                    b.settings_changed = true;
                    d.insert_temp(bridge_id, b);
                }
            });
        }
        ui.ctx().request_repaint();
    }
}

pub(super) fn notify_pause_triggered(ctx: &egui::Context) {
    let bridge_id = scoped_audio_bridge_id();
    ctx.data_mut(|d| {
        if let Some(mut b) = d.get_temp::<ScopedAudioUiBridge>(bridge_id) {
            b.pause_triggered = true;
            d.insert_temp(bridge_id, b);
        }
    });
}

fn render_mute_toggle(
    ui: &mut egui::Ui,
    base_id: egui::Id,
    tool_name: &str,
    enabled: bool,
    min_size: egui::Vec2,
    muted: bool,
) -> bool {
    let label = if muted { "🔇 Muted" } else { "🔊 Audio" };
    let a11y = if muted {
        format!("Unmute {tool_name} sonification audio")
    } else {
        format!("Mute {tool_name} sonification audio")
    };
    let info = egui::WidgetInfo::selected(egui::WidgetType::Button, enabled, !muted, &a11y);
    let resp = ui.add_enabled(enabled, egui::Button::new(label).min_size(min_size));
    resp.widget_info({
        let info = info.clone();
        move || info.clone()
    });
    ui.ctx().data_mut(|d| {
        d.insert_temp(base_id.with("mute_button_info"), info);
    });
    resp.accessible_hover_text(&a11y).clicked()
}

fn render_gain_slider(
    ui: &mut egui::Ui,
    base_id: egui::Id,
    tool_name: &str,
    enabled: bool,
    settings: &mut AcousticSettings,
) -> bool {
    let a11y = format!("Audio gain for {tool_name}");
    let info = egui::WidgetInfo::slider(enabled, f64::from(settings.gain), &a11y);
    let resp = ui.add_enabled(
        enabled,
        egui::Slider::new(&mut settings.gain, 0.0..=1.0).text("Gain"),
    );
    resp.widget_info({
        let info = info.clone();
        move || info.clone()
    });
    ui.ctx().data_mut(|d| {
        d.insert_temp(base_id.with("gain_slider_info"), info);
    });
    resp.accessible_hover_text(&a11y).changed()
}
