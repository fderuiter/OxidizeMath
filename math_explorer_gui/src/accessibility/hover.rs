#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Accessible hover and focus tooltip traits with `Escape` dismissal and polite ARIA live dispatch.

use super::live_region::{announce_status, with_live_region};
use eframe::egui::{self, Response, WidgetText};
use scientific_metadata::theory::TheoryDescribable;

fn resolve_hover_or_focus_dismissal(response: &Response) -> (bool, bool) {
    let is_hovered = if response.enabled() {
        response.hovered()
    } else {
        response.contains_pointer()
    };
    let dismiss_id = response.id.with("tooltip_dismissed");
    let mut dismissed = response
        .ctx
        .data(|d| d.get_temp::<bool>(dismiss_id))
        .unwrap_or(false);

    let escape_pressed = response.ctx.input(|i| i.key_pressed(egui::Key::Escape));
    if escape_pressed && !dismissed && response.lost_focus() {
        response.request_focus();
        dismissed = true;
        response.ctx.data_mut(|d| d.insert_temp(dismiss_id, true));
    }

    let is_focused = response.has_focus() || response.ctx.memory(|mem| mem.has_focus(response.id));
    let is_active = is_hovered || is_focused;
    if is_active {
        if escape_pressed && !dismissed {
            dismissed = true;
            response.ctx.data_mut(|d| d.insert_temp(dismiss_id, true));
        }
    } else if dismissed {
        response.ctx.data_mut(|d| d.insert_temp(dismiss_id, false));
    }
    (is_active, dismissed)
}

fn attach_enabled_or_disabled_hover(response: Response, text: impl Into<WidgetText>) -> Response {
    if response.enabled() {
        response.on_hover_text(text)
    } else {
        response.on_disabled_hover_text(text)
    }
}

fn record_live_tooltip_message(ctx: &egui::Context, message: String) {
    announce_status(&message);
    let active_slot = with_live_region(|r| r.active_slot);
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new("aria_live_message"), message);
        d.insert_temp(egui::Id::new("aria_live_slot"), active_slot);
        d.insert_temp(egui::Id::new("aria_live_dispatched"), true);
    });
}

/// Extension trait attaching keyboard-focusable, `Escape`-dismissible tooltips with polite live status.
pub trait AccessibleHoverText {
    /// Attaches a tooltip announced to polite live regions on pointer hover or keyboard focus.
    fn accessible_hover_text(self, text: impl Into<WidgetText>) -> Self;
}

impl AccessibleHoverText for Response {
    fn accessible_hover_text(self, text: impl Into<WidgetText>) -> Self {
        let (is_active, dismissed) = resolve_hover_or_focus_dismissal(&self);
        if !is_active {
            return attach_enabled_or_disabled_hover(self, text);
        }
        if dismissed {
            return self;
        }
        let widget_text = text.into();
        record_live_tooltip_message(&self.ctx, widget_text.text().to_string());
        attach_enabled_or_disabled_hover(self, widget_text)
    }
}

/// Extension trait attaching `TheoryDescribable` tooltips and phonetic live-region descriptions.
pub trait AccessibleTheoryHover {
    /// Attaches a theory tooltip that stays silent in `SpeechSynthesis` unless user narration is enabled.
    fn accessible_theory_hover(self, theory: &impl TheoryDescribable) -> Self;
}

impl AccessibleTheoryHover for Response {
    fn accessible_theory_hover(self, theory: &impl TheoryDescribable) -> Self {
        let (is_active, dismissed) = resolve_hover_or_focus_dismissal(&self);
        if !is_active {
            if self.ctx.memory(|mem| mem.everything_is_visible()) {
                return attach_enabled_or_disabled_hover(self, theory.theory_description());
            }
            return self;
        }
        if dismissed {
            return self;
        }
        let gained_focus = self.gained_focus();
        let raw_text = theory.theory_description();
        let speech_text = theory.phonetic_description();
        record_live_tooltip_message(&self.ctx, speech_text.clone());
        if gained_focus && super::speech::is_speech_enabled() {
            super::speech::speak_if_enabled(&speech_text);
        }
        attach_enabled_or_disabled_hover(self, raw_text)
    }
}
