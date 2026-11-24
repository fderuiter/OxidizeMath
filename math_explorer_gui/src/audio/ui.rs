//! Accessible high-contrast sonification toolbar and peak `dBFS` / dominant `Hz` visual meter.

// theory_verification!

use eframe::egui;

use super::frame::{
    MeterReading, SonificationFrame, WaveformPreset, FLOOR_DBFS, LIMITER_CEILING_DBFS,
};
use super::wav::{
    MAX_OFFLINE_DURATION_SECS, MIN_OFFLINE_DURATION_SECS, WAV_EXPORT_SAMPLE_RATE,
    WAV_STUDIO_SAMPLE_RATE,
};
use crate::accessibility::{
    paint_canvas_focus_ring, register_canvas_accessibility, AccessibleHoverText,
};
use crate::app::palette::SemanticPalette;

/// Renders the sonification mute/unlock control, timbre preset, offline duration (`1..=15 s`),
/// sample rate (`44.1`/`48 kHz`), `.wav`/`.mid` export buttons, and WCAG AA visual `dBFS` + `Hz` meter.
pub fn render_sonification_panel(
    ui: &mut egui::Ui,
    tool_id: &'static str,
    frame: &SonificationFrame,
    is_paused: bool,
) {
    let bridge_id = crate::framework::scoped_audio_bridge_id();
    let bridge_snap = ui
        .ctx()
        .data(|d| d.get_temp::<crate::framework::ScopedAudioUiBridge>(bridge_id));
    let pitched_frame =
        bridge_snap.map_or(*frame, |b| frame.with_pitch_scale(b.settings.pitch_scale));
    super::submit_tool_frame(tool_id, pitched_frame, is_paused);
    let state = super::snapshot_ui_state();
    let effective_muted = bridge_snap.map_or(state.muted, |b| b.muted);
    let effective_vol = bridge_snap.map_or(state.master_volume, |b| b.settings.gain);

    ui.group(|ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("Sonification").strong());
            render_mute_button(ui, effective_muted);
            render_wav_export_button(ui, tool_id);
            render_midi_export_button(ui, tool_id);
        });

        ui.horizontal_wrapped(|ui| {
            let mut vol_pct = (effective_vol * 100.0).round();
            ui.label("Gain:");
            let slider = egui::Slider::new(&mut vol_pct, 0.0..=100.0)
                .suffix("%")
                .fixed_decimals(0);
            if ui
                .add(slider)
                .accessible_hover_text("Master sonification volume percentage")
                .changed()
            {
                let next_vol = vol_pct * 0.01;
                super::set_user_master_volume(next_vol);
                ui.ctx().data_mut(|d| {
                    if let Some(mut b) =
                        d.get_temp::<crate::framework::ScopedAudioUiBridge>(bridge_id)
                    {
                        b.settings.gain = next_vol;
                        b.settings_changed = true;
                        d.insert_temp(bridge_id, b);
                    }
                });
            }
        });

        render_offline_render_controls(ui, state.offline_duration_secs, state.offline_sample_rate);
        render_timbre_selector(ui, state.waveform_override, frame.waveform);
        render_visual_meter(ui, &state.meter, effective_muted);
    });
}

fn render_mute_button(ui: &mut egui::Ui, is_muted: bool) {
    let label = if is_muted {
        "Enable Audio"
    } else {
        "Mute Audio"
    };
    let tip = if is_muted {
        "Unlock Web Audio and unmute live mathematical sonification"
    } else {
        "Mute live mathematical sonification immediately"
    };
    if ui.button(label).accessible_hover_text(tip).clicked() {
        let next_muted = !is_muted;
        super::set_muted_from_user_gesture_target(next_muted);
        let bridge_id = crate::framework::scoped_audio_bridge_id();
        ui.ctx().data_mut(|d| {
            if let Some(mut b) = d.get_temp::<crate::framework::ScopedAudioUiBridge>(bridge_id) {
                b.muted = next_muted;
                b.toggled_mute = true;
                d.insert_temp(bridge_id, b);
            }
        });
    }
}

fn render_wav_export_button(ui: &mut egui::Ui, tool_id: &'static str) {
    if ui
        .button("Export .wav")
        .accessible_hover_text(
            "Synthesize and download a deterministic 16-bit stereo WAV recording (1s to 15s)",
        )
        .clicked()
    {
        super::export_current_tool_wav(tool_id);
    }
}

fn render_midi_export_button(ui: &mut egui::Ui, tool_id: &'static str) {
    if ui
        .button("Export .mid")
        .accessible_hover_text(
            "Serialize and download a polyphonic SMF Format 1 MIDI score with 14-bit pitch bends",
        )
        .clicked()
    {
        super::export_current_tool_midi(tool_id);
    }
}

fn render_offline_render_controls(ui: &mut egui::Ui, duration_secs: f32, sample_rate: u32) {
    ui.horizontal_wrapped(|ui| {
        let mut dur = duration_secs;
        ui.label("Clip:");
        let drag = egui::DragValue::new(&mut dur)
            .range(MIN_OFFLINE_DURATION_SECS..=MAX_OFFLINE_DURATION_SECS)
            .speed(0.25)
            .suffix(" s")
            .fixed_decimals(1);
        if ui
            .add(drag)
            .accessible_hover_text(
                "Offline WAV and MIDI export duration in seconds (1.0s to 15.0s)",
            )
            .changed()
        {
            super::set_offline_duration_secs(dur);
        }

        let rate_label = if sample_rate >= 46_000 {
            "48 kHz"
        } else {
            "44.1 kHz"
        };
        egui::ComboBox::from_id_salt("sonification_offline_sample_rate")
            .selected_text(rate_label)
            .width(78.0)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(sample_rate < 46_000, "44.1 kHz")
                    .clicked()
                {
                    super::set_offline_sample_rate(WAV_EXPORT_SAMPLE_RATE);
                }
                if ui
                    .selectable_label(sample_rate >= 46_000, "48 kHz")
                    .clicked()
                {
                    super::set_offline_sample_rate(WAV_STUDIO_SAMPLE_RATE);
                }
            });
    });
}

fn render_timbre_selector(
    ui: &mut egui::Ui,
    current_override: Option<WaveformPreset>,
    default_preset: WaveformPreset,
) {
    let active_label = current_override.map_or("Auto (Domain)", WaveformPreset::label);
    ui.horizontal(|ui| {
        ui.label("Timbre:");
        egui::ComboBox::from_id_salt("sonification_timbre_preset")
            .selected_text(active_label)
            .show_ui(ui, |ui| {
                let auto_text = format!("Auto ({})", default_preset.label());
                if ui
                    .selectable_label(current_override.is_none(), auto_text)
                    .clicked()
                {
                    super::set_waveform_override(None);
                }
                for preset in WaveformPreset::ALL {
                    if ui
                        .selectable_label(current_override == Some(preset), preset.label())
                        .clicked()
                    {
                        super::set_waveform_override(Some(preset));
                    }
                }
            });
    });
}

/// Paints the high-contrast peak `dBFS` and dominant `Hz` visual meter with keyboard focus ring.
pub fn render_visual_meter(ui: &mut egui::Ui, meter: &MeterReading, muted: bool) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    let desired = egui::vec2(ui.available_width().max(160.0), 38.0);
    let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::focusable_noninteractive());

    let status_text = format_meter_readout(meter, muted);
    let aria_label = format!("Audio level meter: {status_text}");
    register_canvas_accessibility(
        ui,
        &resp,
        egui::accesskit::Role::Meter,
        &aria_label,
        Some(&status_text),
    );

    if ui.is_rect_visible(rect) {
        let painter = ui.painter();
        painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);

        let bar_area = rect.shrink2(egui::vec2(4.0, 4.0));
        let fill_frac = meter_fill_fraction(meter.peak_dbfs, muted);
        if fill_frac > 0.0 {
            let mut fill_rect = bar_area;
            fill_rect.set_width(bar_area.width() * fill_frac);
            let bar_color = if meter.is_limiter_engaged || meter.peak_dbfs > -6.0 {
                palette.warning
            } else {
                palette.info
            };
            // Blend fill lightly so foreground text retains >= 4.5:1 contrast across all themes.
            let tinted = bar_color.gamma_multiply(0.35);
            painter.rect_filled(fill_rect, 2.0, tinted);
            painter.line_segment(
                [fill_rect.right_top(), fill_rect.right_bottom()],
                egui::Stroke::new(2.0_f32, bar_color),
            );
        }

        let limiter_frac = meter_fill_fraction(LIMITER_CEILING_DBFS, false);
        let lim_x = bar_area.left() + bar_area.width() * limiter_frac;
        painter.line_segment(
            [
                egui::pos2(lim_x, bar_area.top()),
                egui::pos2(lim_x, bar_area.bottom()),
            ],
            egui::Stroke::new(1.5_f32, palette.error),
        );

        let border_color = ui.visuals().widgets.noninteractive.fg_stroke.color;
        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(1.0_f32, border_color),
            egui::epaint::StrokeKind::Inside,
        );
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            &status_text,
            egui::FontId::monospace(11.5),
            ui.visuals().strong_text_color(),
        );
        paint_canvas_focus_ring(ui, &resp);
    }
}

/// Formats peak `dBFS`, dominant `Hz`, and active voice count for visual & ARIA display.
#[must_use]
pub fn format_meter_readout(meter: &MeterReading, muted: bool) -> String {
    if muted {
        format!("MUTED ({FLOOR_DBFS:.1} dBFS) | 0.0 Hz")
    } else {
        format!(
            "{:+.1} dBFS | {:5.1} Hz | {}v",
            meter.peak_dbfs, meter.dominant_hz, meter.active_voices
        )
    }
}

/// Maps `dBFS` in `[-60.0, 0.0]` onto a normalized `[0.0, 1.0]` horizontal bar fraction.
#[inline]
#[must_use]
pub fn meter_fill_fraction(peak_dbfs: f32, muted: bool) -> f32 {
    if muted || !peak_dbfs.is_finite() {
        return 0.0;
    }
    ((peak_dbfs - FLOOR_DBFS) / (0.0 - FLOOR_DBFS)).clamp(0.0, 1.0)
}
