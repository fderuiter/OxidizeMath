//! Offline `.wav` frame rendering for [`crate::tabs::number_theory::prime_spiral::PrimeSpiralWidget`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::PrimeSpiralWidget;
use crate::audio::SonificationFrame;
use crate::framework::InteractiveTool;
use math_explorer::pure_math::number_theory::primes::primes_up_to;

impl PrimeSpiralWidget {
    pub(super) fn render_wav_frames(&self, frame_count: usize) -> Vec<SonificationFrame> {
        let count = frame_count.max(1);
        if !self.has_finite_parameters() {
            return vec![SonificationFrame::silent(); count];
        }
        let size = self.grid_size.max(10);
        let limit = size.saturating_mul(size).max(16);
        let lookup = if self.is_prime_lookup.len() > limit {
            self.is_prime_lookup.clone()
        } else {
            let mut table = vec![false; limit + 1];
            for p in primes_up_to(limit as u64) {
                if (p as usize) < table.len() {
                    table[p as usize] = true;
                }
            }
            table
        };
        let mut local = Self {
            grid_size: size,
            playhead_mode: self.playhead_mode,
            ray_b: self.ray_b,
            ray_c: self.ray_c,
            cursor_step: self.cursor_step,
            tempo_bpm: self.tempo_bpm,
            pulse_gate: true,
            is_prime_lookup: lookup,
            audio_settings: self.audio_settings,
            ..Self::default()
        };
        let mut out = Vec::with_capacity(count);
        for idx in 0..count {
            local.pulse_gate = idx % 4 != 3;
            out.push(local.current_sonification_frame());
            if idx % 4 == 3 {
                local.step_playhead_delta(1);
            }
        }
        out
    }
}
