//! Zero-allocation Ulam prime-gap polyrhythmic interval sonification for [`crate::tabs::number_theory::prime_spiral::PrimeSpiralWidget`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::PrimeSpiralWidget;
use crate::audio::{SonificationFrame, VoiceSpec, WaveformPreset};
use math_explorer::pure_math::number_theory::primes::primes_up_to;
use std::hash::{Hash, Hasher};

#[inline]
fn is_prime_fallback(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) {
        return false;
    }
    let mut d = 5_usize;
    while d.saturating_mul(d) <= n {
        if n.is_multiple_of(d) || n.is_multiple_of(d + 2) {
            return false;
        }
        d += 6;
    }
    true
}

#[inline]
fn lookup_is_prime(lookup: &[bool], n: usize) -> bool {
    if n < lookup.len() {
        lookup[n]
    } else {
        is_prime_fallback(n)
    }
}

#[inline]
fn next_prime_from(lookup: &[bool], start: usize, limit: usize) -> usize {
    let mut cur = start.max(2);
    for _ in 0..512 {
        if lookup_is_prime(lookup, cur) {
            return cur;
        }
        cur = if cur >= limit { 2 } else { cur + 1 };
    }
    2
}

impl PrimeSpiralWidget {
    pub(super) fn ensure_prime_lookup_ready(&mut self) {
        let limit = self.grid_size.max(2).saturating_mul(self.grid_size.max(2));
        if self.is_prime_lookup.len() <= limit {
            self.is_prime_lookup.clear();
            self.is_prime_lookup.resize(limit + 1, false);
            for p in primes_up_to(limit as u64) {
                if (p as usize) < self.is_prime_lookup.len() {
                    self.is_prime_lookup[p as usize] = true;
                }
            }
        }
    }

    pub(super) fn advance_spiral_cursor(&mut self) {
        self.ensure_prime_lookup_ready();
        let limit = self.grid_size.max(2).saturating_mul(self.grid_size.max(2));
        if self.seq_cursor >= limit {
            self.seq_cursor = 2;
        } else {
            self.seq_cursor += 1;
        }
    }

    pub(super) fn compute_state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.grid_size.hash(&mut h);
        self.seq_playing.hash(&mut h);
        self.audio_settings.gain.to_bits().hash(&mut h);
        self.audio_settings.pitch_scale.to_bits().hash(&mut h);
        h.finish().max(1)
    }

    /// Computes a zero-allocation polyrhythmic `SonificationFrame` from Ulam spiral prime gaps.
    #[must_use]
    pub(super) fn compute_sonification_frame(&self) -> SonificationFrame {
        compute_sonification_frame_at(&self.is_prime_lookup, self.grid_size, self.seq_cursor)
    }

    pub(super) fn render_wav_frames(&self, frame_count: usize) -> Vec<SonificationFrame> {
        let count = frame_count.max(1);
        let size = self.grid_size.max(10);
        let limit = size.saturating_mul(size);
        let mut temp_lookup = Vec::new();
        let lookup_ref: &[bool] = if self.is_prime_lookup.len() > limit {
            &self.is_prime_lookup
        } else {
            temp_lookup.resize(limit + 1, false);
            for p in primes_up_to(limit as u64) {
                if (p as usize) < temp_lookup.len() {
                    temp_lookup[p as usize] = true;
                }
            }
            &temp_lookup
        };

        let mut cursor = self.seq_cursor.clamp(2, limit);
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(compute_sonification_frame_at(lookup_ref, size, cursor));
            cursor = if cursor >= limit { 2 } else { cursor + 1 };
        }
        out
    }
}

fn compute_sonification_frame_at(
    lookup: &[bool],
    grid_size: usize,
    seq_cursor: usize,
) -> SonificationFrame {
    let limit = grid_size.max(10).saturating_mul(grid_size.max(10));
    let cursor = seq_cursor.clamp(2, limit);
    let cursor_is_prime = lookup_is_prime(lookup, cursor);

    let p0 = next_prime_from(lookup, cursor, limit);
    let p1 = next_prime_from(lookup, p0.saturating_add(1), limit);
    let p2 = next_prime_from(lookup, p1.saturating_add(1), limit);
    let p3 = next_prime_from(lookup, p2.saturating_add(1), limit);

    let g1 = p1.abs_diff(p0).clamp(1, 36) as f32;
    let g2 = p2.abs_diff(p1).clamp(1, 36) as f32;
    let g3 = p3.abs_diff(p2).clamp(1, 36) as f32;

    let ring = (((cursor as f32).sqrt() + 1.0) * 0.5).floor();
    let root_hz = 165.0 + (ring % 12.0) * 18.0;
    let voice_freqs = [
        root_hz,
        (root_hz * (1.0 + g1 / 12.0)).clamp(110.0, 2_400.0),
        (root_hz * (1.0 + g2 / 8.0)).clamp(110.0, 2_800.0),
        (root_hz * (1.0 + (g1 + g3) / 10.0)).clamp(110.0, 3_200.0),
    ];
    let prime_boost = if cursor_is_prime { 1.0 } else { 0.65 };
    let voice_gains = [
        0.30 * prime_boost,
        0.26 * prime_boost,
        0.22 * prime_boost,
        0.18 * prime_boost,
    ];
    let base_pan = ((cursor as f32) * 0.37).sin().clamp(-0.75, 0.75);
    let cutoff_hz = (1_200.0 + 180.0 * (g1 + g2)).clamp(400.0, 7_200.0);
    let impulse = if cursor_is_prime && (g1 - 2.0).abs() < 0.01 {
        0.70
    } else if cursor_is_prime {
        0.32
    } else {
        0.0
    };

    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::JustHarmonic;
    frame.active_voices = 4;
    frame.master_gain = if cursor_is_prime { 0.74 } else { 0.46 };

    for idx in 0..4 {
        let spread = (idx as f32 - 1.5) * 0.22;
        frame.voices[idx] = VoiceSpec {
            freq_hz: voice_freqs[idx],
            gain: voice_gains[idx],
            pan: (base_pan + spread).clamp(-1.0, 1.0),
            fm_mod_hz: 0.0,
            fm_index: 0.0,
            filter_cutoff_hz: cutoff_hz,
            transient_impulse: if idx == 0 { impulse } else { 0.0 },
        };
    }
    frame.sanitize()
}
