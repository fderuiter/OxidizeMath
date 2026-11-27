//! `f32::tanh` soft-clipping sample limiter and `<= 20 ms` `PlatformInstant` gain ramp envelope.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::web_io::PlatformInstant;

/// Maximum permitted silence ramp duration in milliseconds (`20 ms` acceptance bound).
pub const MAX_SILENCE_RAMP_MS: f32 = 20.0;

/// Target linear gain ramp duration in seconds (`15 ms <= 20 ms`).
pub const SILENCE_RAMP_DURATION_SECS: f32 = 0.015;

/// Soft-clips a raw audio sample into `(-1.0, 1.0)` via `f32::tanh`, zeroing non-finite inputs.
#[inline]
#[must_use]
pub fn soft_clip_sample(sample: f32) -> f32 {
    if !sample.is_finite() {
        return 0.0;
    }
    sample.tanh()
}

/// Soft-clips a stereo `(left, right)` sample pair using `f32::tanh`.
#[inline]
#[must_use]
pub fn soft_clip_stereo(left: f32, right: f32) -> (f32, f32) {
    (soft_clip_sample(left), soft_clip_sample(right))
}

/// Equal-power stereo pan gains `(left_gain, right_gain)` for `pan` in `-1.0..=1.0`.
#[inline]
#[must_use]
pub fn equal_power_pan_gains(pan: f32) -> (f32, f32) {
    let clamped = if pan.is_finite() {
        pan.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let angle = (clamped + 1.0) * core::f32::consts::FRAC_PI_4;
    let (sin_a, cos_a) = angle.sin_cos();
    (cos_a, sin_a)
}

/// Monotonic `PlatformInstant`-driven gain ramp guaranteeing `<= 20 ms` fade-out to silence.
#[derive(Clone, Debug)]
pub struct GainRampEnvelope {
    start_gain: f32,
    current_gain: f32,
    target_gain: f32,
    ramp_started_at: PlatformInstant,
    ramp_duration_secs: f32,
}

impl Default for GainRampEnvelope {
    fn default() -> Self {
        Self::new_muted()
    }
}

impl GainRampEnvelope {
    /// Creates an envelope initialized at `0.0` gain (muted by default).
    #[must_use]
    pub fn new_muted() -> Self {
        Self {
            start_gain: 0.0,
            current_gain: 0.0,
            target_gain: 0.0,
            ramp_started_at: PlatformInstant::now(),
            ramp_duration_secs: SILENCE_RAMP_DURATION_SECS,
        }
    }

    /// Initiates a smooth linear gain transition toward `target` over `duration_secs` (`<= 15 ms`).
    pub fn ramp_to(&mut self, target: f32) {
        let now = PlatformInstant::now();
        self.current_gain = self.sample_at(now);
        let safe_target = if target.is_finite() {
            target.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.start_gain = self.current_gain;
        self.target_gain = safe_target;
        self.ramp_started_at = now;
        self.ramp_duration_secs = SILENCE_RAMP_DURATION_SECS;
    }

    /// Initiates an immediate `15 ms` (`<= 20 ms`) ramp to `0.0` silence.
    pub fn ramp_to_zero(&mut self) {
        self.ramp_to(0.0);
    }

    /// Evaluates the interpolated gain at `now` using monotonic `PlatformInstant` elapsed time.
    #[must_use]
    pub fn sample_at(&self, now: PlatformInstant) -> f32 {
        self.sample_after_elapsed(now.duration_since(self.ramp_started_at))
    }

    /// Evaluates the interpolated gain after a given `elapsed` duration from `ramp_started_at`.
    #[must_use]
    pub fn sample_after_elapsed(&self, elapsed_dur: std::time::Duration) -> f32 {
        if (self.target_gain - self.start_gain).abs() < 1e-7 || self.ramp_duration_secs <= 0.0 {
            return self.target_gain;
        }
        let elapsed = elapsed_dur.as_secs_f32();
        if elapsed >= self.ramp_duration_secs {
            self.target_gain
        } else {
            let alpha = (elapsed / self.ramp_duration_secs).clamp(0.0, 1.0);
            self.start_gain + (self.target_gain - self.start_gain) * alpha
        }
    }

    /// Updates and returns the current gain at `PlatformInstant::now()`.
    pub fn advance(&mut self) -> f32 {
        let g = self.sample_at(PlatformInstant::now());
        self.current_gain = g;
        g
    }

    /// Returns the target gain at the end of the active ramp.
    #[must_use]
    pub const fn target_gain(&self) -> f32 {
        self.target_gain
    }

    /// Returns the timestamp when the most recent ramp began.
    #[must_use]
    pub const fn ramp_started_at(&self) -> PlatformInstant {
        self.ramp_started_at
    }
}
