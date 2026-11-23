//! Zero-dependency Standard MIDI File (SMF Format 1) serializer with 14-bit microtonal pitch bends.
//!
//! Encodes polyphonic [`SonificationFrame`] timelines and discrete voice note spans into multi-track
//! SMF Format 1 `.mid` binaries (`MThd` + tempo `MTrk` + per-voice `MTrk`s on channels `0..8`,
//! avoiding General MIDI percussion channel `9`). Pitch bends use a `±2`-semitone RPN `00 00`
//! sensitivity window (`0.0244` cents/LSB resolution, well within `<= 0.5` cents).

// theory_verification!

use super::frame::{clamp_freq_hz, sanitize_f32, SonificationFrame, MAX_VOICES};
use super::wav::MAX_OFFLINE_DURATION_SECS;

/// Pulses (ticks) per quarter note in exported SMF Format 1 headers.
pub const MIDI_TICKS_PER_QUARTER: u16 = 480;

/// Conductor tempo in microseconds per quarter note (`500,000 us` = `120 BPM`, `960 ticks/s`).
pub const MIDI_TEMPO_US_PER_QUARTER: u32 = 500_000;

/// Pitch-bend range in semitones configured via RPN `00 00` (`±2` semitones = `±200` cents).
pub const MIDI_PITCH_BEND_SEMITONES: f64 = 2.0;

/// Center (unbent) 14-bit MIDI pitch-bend value (`0x2000 = 8192`).
pub const MIDI_PITCH_BEND_CENTER: u16 = 8_192;

/// Non-percussion MIDI channels (`0..8`) assigned to `SonificationFrame` voices `0..MAX_VOICES`.
pub const VOICE_MIDI_CHANNELS: [u8; MAX_VOICES] = [0, 1, 2, 3, 4, 5, 6, 7];

/// Quantized 12-TET MIDI key plus 14-bit microtonal pitch bend relative to `A4 = 440.0 Hz`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MidiPitch {
    /// Nearest 7-bit MIDI note number (`0..=127`).
    pub note: u8,
    /// Unsigned 14-bit pitch-bend value (`0..=16383`, `8192` = center) for `±2` semitones.
    pub bend_14bit: u16,
}

impl MidiPitch {
    /// Maps an audible frequency in `Hz` (`20.0..=8000.0`) to `(note, bend_14bit)` within `<= 0.02` cents.
    #[must_use]
    pub fn from_freq_hz(freq_hz: f32) -> Self {
        let hz = f64::from(clamp_freq_hz(freq_hz));
        let continuous = 69.0 + 12.0 * (hz / 440.0).log2();
        let note = continuous.round().clamp(0.0, 127.0) as u8;
        let delta = (continuous - f64::from(note))
            .clamp(-MIDI_PITCH_BEND_SEMITONES, MIDI_PITCH_BEND_SEMITONES);
        let raw_bend =
            f64::from(MIDI_PITCH_BEND_CENTER) + (delta / MIDI_PITCH_BEND_SEMITONES) * 8_192.0;
        let bend_14bit = raw_bend.round().clamp(0.0, 16_383.0) as u16;
        Self { note, bend_14bit }
    }

    /// Reconstructs the exact acoustic frequency in `Hz` represented by `(note, bend_14bit)`.
    #[must_use]
    pub fn to_freq_hz(self) -> f64 {
        let bend_semis =
            MIDI_PITCH_BEND_SEMITONES * (f64::from(self.bend_14bit) - 8_192.0) / 8_192.0;
        440.0 * ((f64::from(self.note) + bend_semis - 69.0) / 12.0).exp2()
    }

    /// Computes absolute pitch deviation in cents (`1200 * |log2(f_midi / f_target)|`).
    #[must_use]
    pub fn cents_error(self, target_hz: f64) -> f64 {
        let denom = if target_hz.is_finite() && target_hz > 1.0e-9 {
            target_hz
        } else {
            440.0
        };
        1_200.0 * (self.to_freq_hz() / denom).log2().abs()
    }
}

/// Discrete microtonal note event scheduled on a single monophonic voice channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MidiNoteSpan {
    /// Start position in MIDI ticks (`960` ticks/second at `120 BPM`).
    pub start_tick: u32,
    /// Non-zero gate duration in MIDI ticks.
    pub duration_ticks: u32,
    /// Target fundamental frequency in `Hz` (`20.0..=8000.0`).
    pub freq_hz: f32,
    /// Attack velocity (`1..=127`).
    pub velocity: u8,
    /// Stereo pan (`-1.0` left to `+1.0` right).
    pub pan: f32,
}

/// Converts seconds (`0.0..=15.0`) to SMF ticks (`960` ticks/second at `480 TPQ`, `120 BPM`).
#[inline]
#[must_use]
pub fn seconds_to_ticks(secs: f64) -> u32 {
    let clamped = if secs.is_finite() {
        secs.clamp(0.0, f64::from(MAX_OFFLINE_DURATION_SECS))
    } else {
        0.0
    };
    let ticks_per_sec =
        f64::from(MIDI_TICKS_PER_QUARTER) * (1_000_000.0 / f64::from(MIDI_TEMPO_US_PER_QUARTER));
    (clamped * ticks_per_sec).round() as u32
}

/// Appends a Standard MIDI variable-length quantity (`VLQ`, 1 to 4 bytes, 7 bits per byte).
pub fn encode_vlq(mut value: u32, out: &mut Vec<u8>) {
    value &= 0x0FFF_FFFF;
    let mut buf = [0_u8; 4];
    let mut idx = 3_usize;
    buf[idx] = (value & 0x7F) as u8;
    value >>= 7;
    while value > 0 && idx > 0 {
        idx -= 1;
        buf[idx] = ((value & 0x7F) as u8) | 0x80;
        value >>= 7;
    }
    out.extend_from_slice(&buf[idx..]);
}

/// Decodes a Standard MIDI variable-length quantity from `bytes`, returning `(value, bytes_read)`.
#[must_use]
pub fn decode_vlq(bytes: &[u8]) -> Option<(u32, usize)> {
    let mut val = 0_u32;
    for (i, &b) in bytes.iter().take(4).enumerate() {
        val = (val << 7) | u32::from(b & 0x7F);
        if b & 0x80 == 0 {
            return Some((val, i + 1));
        }
    }
    None
}

/// Serializes up to `MAX_VOICES` polyphonic voice note-span lists into an SMF Format 1 `.mid` file.
#[must_use]
pub fn encode_midi_voice_tracks(title: &str, voice_tracks: &[Vec<MidiNoteSpan>]) -> Vec<u8> {
    let mut tracks: Vec<Vec<u8>> = Vec::with_capacity(1 + MAX_VOICES);
    tracks.push(build_conductor_track(title));

    for (voice_idx, spans) in voice_tracks.iter().take(MAX_VOICES).enumerate() {
        if spans.is_empty() {
            continue;
        }
        let channel = VOICE_MIDI_CHANNELS[voice_idx];
        let track_name = format!("Voice {}", voice_idx + 1);
        tracks.push(build_span_voice_track(&track_name, channel, spans));
    }
    if tracks.len() == 1 {
        tracks.push(build_span_voice_track(
            "Voice 1",
            VOICE_MIDI_CHANNELS[0],
            &[],
        ));
    }
    assemble_smf_format1(&tracks)
}

/// Serializes a sequence of [`SonificationFrame`] snapshots into a smooth SMF Format 1 `.mid` file.
#[must_use]
pub fn encode_frames_to_midi(
    title: &str,
    frames: &[SonificationFrame],
    frame_dt_secs: f64,
) -> Vec<u8> {
    let dt_ticks = seconds_to_ticks(frame_dt_secs.max(1.0 / 240.0)).max(1);
    let max_ticks = seconds_to_ticks(f64::from(MAX_OFFLINE_DURATION_SECS));
    let mut tracks: Vec<Vec<u8>> = Vec::with_capacity(1 + MAX_VOICES);
    tracks.push(build_conductor_track(title));

    for (voice_idx, &channel) in VOICE_MIDI_CHANNELS.iter().enumerate() {
        let body = build_frame_voice_track(frames, voice_idx, channel, dt_ticks, max_ticks);
        if let Some(track) = body {
            tracks.push(track);
        }
    }
    if tracks.len() == 1 {
        tracks.push(build_span_voice_track(
            "Voice 1",
            VOICE_MIDI_CHANNELS[0],
            &[],
        ));
    }
    assemble_smf_format1(&tracks)
}

fn assemble_smf_format1(tracks: &[Vec<u8>]) -> Vec<u8> {
    let body_bytes: usize = tracks.iter().map(|t| 8 + t.len()).sum();
    let mut out = Vec::with_capacity(14 + body_bytes);
    out.extend_from_slice(b"MThd");
    out.extend_from_slice(&6_u32.to_be_bytes());
    out.extend_from_slice(&1_u16.to_be_bytes());
    out.extend_from_slice(&(tracks.len().min(u16::MAX as usize) as u16).to_be_bytes());
    out.extend_from_slice(&MIDI_TICKS_PER_QUARTER.to_be_bytes());

    for track in tracks {
        out.extend_from_slice(b"MTrk");
        out.extend_from_slice(&(track.len() as u32).to_be_bytes());
        out.extend_from_slice(track);
    }
    out
}

fn build_conductor_track(title: &str) -> Vec<u8> {
    let mut trk = Vec::with_capacity(64 + title.len());
    push_track_name(&mut trk, title);
    // Tempo Meta Event: FF 51 03 tt tt tt
    encode_vlq(0, &mut trk);
    let t_bytes = MIDI_TEMPO_US_PER_QUARTER.to_be_bytes();
    trk.extend_from_slice(&[0xFF, 0x51, 0x03, t_bytes[1], t_bytes[2], t_bytes[3]]);
    // 4/4 Time Signature: FF 58 04 04 02 18 08
    encode_vlq(0, &mut trk);
    trk.extend_from_slice(&[0xFF, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08]);
    push_end_of_track(&mut trk);
    trk
}

fn push_track_name(trk: &mut Vec<u8>, name: &str) {
    let clean: Vec<u8> = name
        .bytes()
        .filter(|b| b.is_ascii_graphic() || *b == b' ')
        .take(64)
        .collect();
    encode_vlq(0, trk);
    trk.extend_from_slice(&[0xFF, 0x03]);
    encode_vlq(clean.len() as u32, trk);
    trk.extend_from_slice(&clean);
}

fn push_pitch_bend_range_rpn(trk: &mut Vec<u8>, channel: u8) {
    let cc = 0xB0 | (channel & 0x0F);
    let semitones = MIDI_PITCH_BEND_SEMITONES.round() as u8;
    for (controller, val) in [
        (101, 0),
        (100, 0),
        (6, semitones),
        (38, 0),
        (101, 127),
        (100, 127),
    ] {
        encode_vlq(0, trk);
        trk.extend_from_slice(&[cc, controller, val]);
    }
}

fn push_pitch_bend(trk: &mut Vec<u8>, delta: u32, channel: u8, bend_14bit: u16) {
    let clamped = bend_14bit.min(16_383);
    encode_vlq(delta, trk);
    trk.push(0xE0 | (channel & 0x0F));
    trk.push((clamped & 0x7F) as u8);
    trk.push(((clamped >> 7) & 0x7F) as u8);
}

fn push_cc(trk: &mut Vec<u8>, delta: u32, channel: u8, cc_num: u8, value: u8) {
    encode_vlq(delta, trk);
    trk.extend_from_slice(&[0xB0 | (channel & 0x0F), cc_num & 0x7F, value & 0x7F]);
}

fn push_end_of_track(trk: &mut Vec<u8>) {
    encode_vlq(0, trk);
    trk.extend_from_slice(&[0xFF, 0x2F, 0x00]);
}

fn pan_to_midi_cc10(pan: f32) -> u8 {
    let clean = sanitize_f32(pan, 0.0, -1.0, 1.0);
    ((clean + 1.0) * 63.5).round().clamp(0.0, 127.0) as u8
}

fn build_span_voice_track(name: &str, channel: u8, spans: &[MidiNoteSpan]) -> Vec<u8> {
    let max_ticks = seconds_to_ticks(f64::from(MAX_OFFLINE_DURATION_SECS));
    let mut sorted: Vec<MidiNoteSpan> = spans
        .iter()
        .copied()
        .filter(|s| s.duration_ticks > 0 && s.velocity > 0 && s.start_tick < max_ticks)
        .collect();
    sorted.sort_by_key(|s| s.start_tick);

    let mut trk = Vec::with_capacity(64 + sorted.len() * 20);
    push_track_name(&mut trk, name);
    push_pitch_bend_range_rpn(&mut trk, channel);

    let mut cursor = 0_u32;
    for span in sorted {
        let start = span.start_tick.max(cursor).min(max_ticks);
        let end = start.saturating_add(span.duration_ticks).min(max_ticks);
        if end <= start {
            continue;
        }
        let pitch = MidiPitch::from_freq_hz(span.freq_hz);
        push_cc(
            &mut trk,
            start - cursor,
            channel,
            10,
            pan_to_midi_cc10(span.pan),
        );
        push_pitch_bend(&mut trk, 0, channel, pitch.bend_14bit);
        encode_vlq(0, &mut trk);
        trk.extend_from_slice(&[
            0x90 | (channel & 0x0F),
            pitch.note,
            span.velocity.clamp(1, 127),
        ]);
        encode_vlq(end - start, &mut trk);
        trk.extend_from_slice(&[0x80 | (channel & 0x0F), pitch.note, 0]);
        cursor = end;
    }
    push_end_of_track(&mut trk);
    trk
}

#[derive(Clone, Copy)]
struct ActiveVoiceState {
    note: u8,
    bend: u16,
    expr: u8,
    pan: u8,
}

struct FrameVoiceWriter {
    trk: Vec<u8>,
    voice_idx: usize,
    channel: u8,
    last_event_tick: u32,
    active: Option<ActiveVoiceState>,
    emitted_any: bool,
}

impl FrameVoiceWriter {
    fn new(frame_count: usize, voice_idx: usize, channel: u8) -> Self {
        let mut trk = Vec::with_capacity(frame_count * 8 + 64);
        push_track_name(&mut trk, &format!("Voice {}", voice_idx + 1));
        push_pitch_bend_range_rpn(&mut trk, channel);
        Self {
            trk,
            voice_idx,
            channel,
            last_event_tick: 0,
            active: None,
            emitted_any: false,
        }
    }

    fn step_frame(&mut self, frame: SonificationFrame, tick: u32) {
        let voice = frame.voices[self.voice_idx];
        let amp = voice.gain * frame.master_gain;
        if self.voice_idx >= frame.active_voices as usize || amp <= 0.015 {
            if let Some(prev) = self.active.take() {
                encode_vlq(tick - self.last_event_tick, &mut self.trk);
                self.trk
                    .extend_from_slice(&[0x80 | (self.channel & 0x0F), prev.note, 0]);
                self.last_event_tick = tick;
            }
            return;
        }
        let pitch = MidiPitch::from_freq_hz(voice.freq_hz);
        let expr = (amp * 127.0).round().clamp(16.0, 127.0) as u8;
        let pan = pan_to_midi_cc10(voice.pan);
        if let Some(mut state) = self.active {
            if state.note == pitch.note && voice.transient_impulse <= 0.25 {
                self.emit_continuous_updates(tick, &mut state, pitch.bend_14bit, expr, pan);
                self.active = Some(state);
                return;
            }
        }
        self.retrigger_voice(tick, pitch, expr, pan, amp, voice.transient_impulse);
        self.emitted_any = true;
    }

    fn retrigger_voice(
        &mut self,
        tick: u32,
        pitch: MidiPitch,
        expr: u8,
        pan: u8,
        amp: f32,
        transient_impulse: f32,
    ) {
        let mut delta = tick - self.last_event_tick;
        if let Some(prev) = self.active.take() {
            encode_vlq(delta, &mut self.trk);
            self.trk
                .extend_from_slice(&[0x80 | (self.channel & 0x0F), prev.note, 0]);
            delta = 0;
        }
        push_cc(&mut self.trk, delta, self.channel, 10, pan);
        push_cc(&mut self.trk, 0, self.channel, 11, expr);
        push_pitch_bend(&mut self.trk, 0, self.channel, pitch.bend_14bit);
        let vel = ((amp + 0.25 * transient_impulse) * 120.0)
            .round()
            .clamp(24.0, 127.0) as u8;
        encode_vlq(0, &mut self.trk);
        self.trk
            .extend_from_slice(&[0x90 | (self.channel & 0x0F), pitch.note, vel]);
        self.last_event_tick = tick;
        self.active = Some(ActiveVoiceState {
            note: pitch.note,
            bend: pitch.bend_14bit,
            expr,
            pan,
        });
    }

    fn emit_continuous_updates(
        &mut self,
        tick: u32,
        state: &mut ActiveVoiceState,
        bend: u16,
        expr: u8,
        pan: u8,
    ) {
        if state.bend != bend {
            push_pitch_bend(
                &mut self.trk,
                tick - self.last_event_tick,
                self.channel,
                bend,
            );
            self.last_event_tick = tick;
            state.bend = bend;
        }
        if state.expr.abs_diff(expr) >= 2 {
            push_cc(
                &mut self.trk,
                tick - self.last_event_tick,
                self.channel,
                11,
                expr,
            );
            self.last_event_tick = tick;
            state.expr = expr;
        }
        if state.pan.abs_diff(pan) >= 2 {
            push_cc(
                &mut self.trk,
                tick - self.last_event_tick,
                self.channel,
                10,
                pan,
            );
            self.last_event_tick = tick;
            state.pan = pan;
        }
    }

    fn finish(mut self, total_frames: usize, dt_ticks: u32, max_ticks: u32) -> Option<Vec<u8>> {
        if let Some(prev) = self.active {
            let end_tick = ((total_frames as u32).saturating_mul(dt_ticks))
                .min(max_ticks)
                .max(self.last_event_tick + 1);
            encode_vlq(end_tick - self.last_event_tick, &mut self.trk);
            self.trk
                .extend_from_slice(&[0x80 | (self.channel & 0x0F), prev.note, 0]);
        }
        if !self.emitted_any {
            return None;
        }
        push_end_of_track(&mut self.trk);
        Some(self.trk)
    }
}

fn build_frame_voice_track(
    frames: &[SonificationFrame],
    voice_idx: usize,
    channel: u8,
    dt_ticks: u32,
    max_ticks: u32,
) -> Option<Vec<u8>> {
    let mut writer = FrameVoiceWriter::new(frames.len(), voice_idx, channel);
    for (idx, raw_frame) in frames.iter().enumerate() {
        let tick = (idx as u32).saturating_mul(dt_ticks);
        if tick >= max_ticks {
            break;
        }
        writer.step_frame(raw_frame.sanitize(), tick);
    }
    writer.finish(frames.len(), dt_ticks, max_ticks)
}
