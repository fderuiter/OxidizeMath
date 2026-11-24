//! User-gesture-gated studio preview controller for stereo `AudioBufferSourceNode` impulse
//! response strikes and morphable `PeriodicWave` wavetable audition under strict `'self'` CSP.

use std::cell::RefCell;

use super::backend::MuteReason;
use super::frame::{clamp_freq_hz, sanitize_f32};
#[cfg(target_arch = "wasm32")]
use super::frame::{
    LIMITER_CEILING_DBFS, LIMITER_CEILING_LINEAR, MUTE_RAMP_TAU_SEC, PARAM_SMOOTH_TAU_SEC,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;
#[cfg(target_arch = "wasm32")]
use web_sys::{
    AudioBuffer, AudioBufferSourceNode, AudioContext, AudioContextState, AudioDestinationNode,
    AudioScheduledSourceNode, DynamicsCompressorNode, GainNode, OscillatorNode, OscillatorType,
    PeriodicWave, PeriodicWaveOptions,
};

/// Immutable snapshot of studio preview audition state for UI controls and contract tests.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StudioPreviewSnapshot {
    /// True once an explicit user gesture has unlocked studio preview audio.
    pub unlocked: bool,
    /// True while continuous morphable `PeriodicWave` wavetable audition is playing.
    pub wavetable_auditioning: bool,
    /// Current audition oscillator pitch in Hz.
    pub audition_freq_hz: f32,
    /// Number of stereo frames loaded in the most recent IR strike audition.
    pub last_strike_frames: usize,
    /// Sample rate (Hz) of the most recent IR strike audition.
    pub last_strike_sample_rate: u32,
    /// Last lifecycle suspension reason.
    pub last_mute_reason: MuteReason,
}

#[cfg(target_arch = "wasm32")]
struct WasmStudioGraph {
    ctx: AudioContext,
    strike_gain: GainNode,
    active_strike: Option<AudioBufferSourceNode>,
    wt_osc: OscillatorNode,
    wt_gain: GainNode,
    _compressor: DynamicsCompressorNode,
    _ceiling: GainNode,
    _destination: AudioDestinationNode,
}

/// Dual-target studio preview controller (Web Audio on `wasm32`, deterministic state on host).
pub struct StudioPreviewController {
    #[cfg(target_arch = "wasm32")]
    graph: Option<WasmStudioGraph>,
    unlocked: bool,
    wavetable_auditioning: bool,
    audition_freq_hz: f32,
    last_strike_frames: usize,
    last_strike_sample_rate: u32,
    last_mute_reason: MuteReason,
}

impl Default for StudioPreviewController {
    fn default() -> Self {
        Self::new()
    }
}

impl StudioPreviewController {
    /// Creates an idle, locked preview controller with zero `AudioContext` instances.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            #[cfg(target_arch = "wasm32")]
            graph: None,
            unlocked: false,
            wavetable_auditioning: false,
            audition_freq_hz: 110.0,
            last_strike_frames: 0,
            last_strike_sample_rate: 0,
            last_mute_reason: MuteReason::InitialDefault,
        }
    }

    /// Returns an immutable state snapshot.
    #[must_use]
    pub const fn snapshot(&self) -> StudioPreviewSnapshot {
        StudioPreviewSnapshot {
            unlocked: self.unlocked,
            wavetable_auditioning: self.wavetable_auditioning,
            audition_freq_hz: self.audition_freq_hz,
            last_strike_frames: self.last_strike_frames,
            last_strike_sample_rate: self.last_strike_sample_rate,
            last_mute_reason: self.last_mute_reason,
        }
    }

    /// Returns `true` on `wasm32` only after an explicit user gesture instantiated `AudioContext`.
    #[must_use]
    pub const fn has_audio_context(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            self.graph.is_some()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.unlocked
        }
    }

    /// Triggers one-shot stereo IR playback via `AudioBufferSourceNode` strictly from a user gesture.
    pub fn trigger_ir_strike_from_user_gesture(
        &mut self,
        sample_rate_hz: u32,
        left: &[f32],
        right: &[f32],
        volume: f32,
    ) -> Result<(), String> {
        let frames = left.len().min(right.len());
        if frames == 0 {
            return Err("IR buffer is empty".to_string());
        }
        let sr = sample_rate_hz.clamp(8_000, 96_000);
        let gain = sanitize_f32(volume, 0.85, 0.0, 1.0);

        #[cfg(target_arch = "wasm32")]
        {
            self.ensure_wasm_graph_from_gesture()?;
            if let Some(graph) = &mut self.graph {
                play_wasm_stereo_buffer(graph, sr, &left[..frames], &right[..frames], gain)?;
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = gain;
        }

        self.unlocked = true;
        self.last_strike_frames = frames;
        self.last_strike_sample_rate = sr;
        Ok(())
    }

    /// Toggles continuous wavetable `PeriodicWave` audition strictly from a user button gesture.
    pub fn toggle_wavetable_audition_from_user_gesture(
        &mut self,
        real: &[f32],
        imag: &[f32],
        freq_hz: f32,
        volume: f32,
    ) -> Result<bool, String> {
        if self.wavetable_auditioning {
            self.stop_wavetable_audition();
            return Ok(false);
        }
        let pitch = clamp_freq_hz(freq_hz);
        let gain = sanitize_f32(volume, 0.65, 0.0, 1.0);

        #[cfg(target_arch = "wasm32")]
        {
            self.ensure_wasm_graph_from_gesture()?;
            if let Some(graph) = &mut self.graph {
                apply_wasm_periodic_wave(graph, real, imag, pitch, gain)?;
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if real.is_empty() || imag.is_empty() {
                return Err("PeriodicWave harmonic arrays must not be empty".to_string());
            }
            let _ = gain;
        }

        self.unlocked = true;
        self.wavetable_auditioning = true;
        self.audition_freq_hz = pitch;
        Ok(true)
    }

    /// Updates morph `PeriodicWave` partials and pitch only if audition was already unlocked by gesture.
    pub fn update_wavetable_audition(
        &mut self,
        real: &[f32],
        imag: &[f32],
        freq_hz: f32,
        volume: f32,
    ) -> Result<(), String> {
        let pitch = clamp_freq_hz(freq_hz);
        self.audition_freq_hz = pitch;
        if !self.unlocked || !self.wavetable_auditioning {
            return Ok(());
        }
        let gain = sanitize_f32(volume, 0.65, 0.0, 1.0);
        #[cfg(target_arch = "wasm32")]
        if let Some(graph) = &mut self.graph {
            apply_wasm_periodic_wave(graph, real, imag, pitch, gain)?;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (real, imag, gain);
        }
        Ok(())
    }

    /// Stops wavetable oscillator audition cleanly without destroying the audio graph.
    pub fn stop_wavetable_audition(&mut self) {
        self.wavetable_auditioning = false;
        #[cfg(target_arch = "wasm32")]
        if let Some(graph) = &self.graph {
            let now = graph.ctx.current_time();
            let _ = graph
                .wt_gain
                .gain()
                .set_target_at_time(0.0, now, MUTE_RAMP_TAU_SEC);
        }
    }

    /// Suspends all strike and wavetable audition output within `<= 30 ms`.
    pub fn suspend_all(&mut self, reason: MuteReason) {
        self.wavetable_auditioning = false;
        self.unlocked = false;
        self.last_mute_reason = reason;
        #[cfg(target_arch = "wasm32")]
        if let Some(graph) = &mut self.graph {
            let now = graph.ctx.current_time();
            let _ = graph
                .wt_gain
                .gain()
                .set_target_at_time(0.0, now, MUTE_RAMP_TAU_SEC);
            let _ = graph
                .strike_gain
                .gain()
                .set_target_at_time(0.0, now, MUTE_RAMP_TAU_SEC);
            if let Some(prev) = graph.active_strike.take() {
                let _ = AudioScheduledSourceNode::stop(&prev);
                let _ = prev.disconnect();
            }
            if graph.ctx.state() == AudioContextState::Running {
                let _ = graph.ctx.suspend();
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn ensure_wasm_graph_from_gesture(&mut self) -> Result<(), String> {
        if self.graph.is_none() {
            self.graph = Some(build_wasm_studio_graph()?);
        }
        if let Some(graph) = &self.graph {
            if graph.ctx.state() == AudioContextState::Suspended {
                let _ = graph.ctx.resume().map_err(format_js_err)?;
            }
        }
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
fn build_wasm_studio_graph() -> Result<WasmStudioGraph, String> {
    let ctx = AudioContext::new().map_err(format_js_err)?;
    let compressor = ctx.create_dynamics_compressor().map_err(format_js_err)?;
    compressor.threshold().set_value(LIMITER_CEILING_DBFS);
    compressor.knee().set_value(0.0);
    compressor.ratio().set_value(20.0);
    compressor.attack().set_value(0.001);
    compressor.release().set_value(0.050);

    let ceiling = ctx.create_gain().map_err(format_js_err)?;
    ceiling.gain().set_value(LIMITER_CEILING_LINEAR);
    let destination = ctx.destination();

    let strike_gain = ctx.create_gain().map_err(format_js_err)?;
    strike_gain.gain().set_value(0.85);

    let wt_osc = ctx.create_oscillator().map_err(format_js_err)?;
    wt_osc.set_type(OscillatorType::Sine);
    wt_osc.frequency().set_value(110.0);
    let wt_gain = ctx.create_gain().map_err(format_js_err)?;
    wt_gain.gain().set_value(0.0);

    wt_osc
        .connect_with_audio_node(&wt_gain)
        .map_err(format_js_err)?;
    wt_gain
        .connect_with_audio_node(&compressor)
        .map_err(format_js_err)?;
    strike_gain
        .connect_with_audio_node(&compressor)
        .map_err(format_js_err)?;
    compressor
        .connect_with_audio_node(&ceiling)
        .map_err(format_js_err)?;
    ceiling
        .connect_with_audio_node(&destination)
        .map_err(format_js_err)?;
    wt_osc.start().map_err(format_js_err)?;

    Ok(WasmStudioGraph {
        ctx,
        strike_gain,
        active_strike: None,
        wt_osc,
        wt_gain,
        _compressor: compressor,
        _ceiling: ceiling,
        _destination: destination,
    })
}

#[cfg(target_arch = "wasm32")]
fn play_wasm_stereo_buffer(
    graph: &mut WasmStudioGraph,
    sample_rate_hz: u32,
    left: &[f32],
    right: &[f32],
    gain: f32,
) -> Result<(), String> {
    if let Some(prev) = graph.active_strike.take() {
        let _ = AudioScheduledSourceNode::stop(&prev);
        let _ = prev.disconnect();
    }
    let frames = left.len().min(right.len());
    let buffer: AudioBuffer = graph
        .ctx
        .create_buffer(2, frames as u32, sample_rate_hz as f32)
        .map_err(format_js_err)?;
    let clean_l: Vec<f32> = left[..frames]
        .iter()
        .map(|&s| {
            if s.is_finite() {
                s.clamp(-1.0, 1.0)
            } else {
                0.0
            }
        })
        .collect();
    let clean_r: Vec<f32> = right[..frames]
        .iter()
        .map(|&s| {
            if s.is_finite() {
                s.clamp(-1.0, 1.0)
            } else {
                0.0
            }
        })
        .collect();
    buffer.copy_to_channel(&clean_l, 0).map_err(format_js_err)?;
    buffer.copy_to_channel(&clean_r, 1).map_err(format_js_err)?;

    let now = graph.ctx.current_time();
    let _ = graph
        .strike_gain
        .gain()
        .set_target_at_time(gain, now, 0.002);
    let source = graph.ctx.create_buffer_source().map_err(format_js_err)?;
    source.set_buffer(Some(&buffer));
    source
        .connect_with_audio_node(&graph.strike_gain)
        .map_err(format_js_err)?;
    source.start().map_err(format_js_err)?;
    graph.active_strike = Some(source);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn apply_wasm_periodic_wave(
    graph: &mut WasmStudioGraph,
    real: &[f32],
    imag: &[f32],
    freq_hz: f32,
    gain: f32,
) -> Result<(), String> {
    let len = real.len().min(imag.len());
    if len < 2 {
        return Err("PeriodicWave requires at least 2 harmonic entries".to_string());
    }
    let real_arr = js_sys::Float32Array::from(&real[..len]);
    let imag_arr = js_sys::Float32Array::from(&imag[..len]);
    let opts = PeriodicWaveOptions::new();
    opts.set_real(real_arr.as_ref());
    opts.set_imag(imag_arr.as_ref());
    let wave = PeriodicWave::new_with_options(&graph.ctx, &opts).map_err(format_js_err)?;
    graph.wt_osc.set_periodic_wave(&wave);
    let now = graph.ctx.current_time();
    graph
        .wt_osc
        .frequency()
        .set_target_at_time(freq_hz, now, PARAM_SMOOTH_TAU_SEC)
        .map_err(format_js_err)?;
    graph
        .wt_gain
        .gain()
        .set_target_at_time(gain * 0.45, now, PARAM_SMOOTH_TAU_SEC)
        .map_err(format_js_err)?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn format_js_err(err: JsValue) -> String {
    err.as_string()
        .unwrap_or_else(|| format!("Studio Web Audio error: {err:?}"))
}

thread_local! {
    static STUDIO_PREVIEW: RefCell<StudioPreviewController> =
        const { RefCell::new(StudioPreviewController::new()) };
}

/// Reads a snapshot of the thread-local studio preview controller.
#[must_use]
pub fn snapshot_studio_preview() -> StudioPreviewSnapshot {
    STUDIO_PREVIEW.with(|cell| cell.borrow().snapshot())
}

/// Plays a stereo impulse-response buffer through the thread-local preview graph on user gesture.
pub fn trigger_studio_ir_strike_from_gesture(
    sample_rate_hz: u32,
    left: &[f32],
    right: &[f32],
    volume: f32,
) -> Result<(), String> {
    STUDIO_PREVIEW.with(|cell| {
        cell.borrow_mut()
            .trigger_ir_strike_from_user_gesture(sample_rate_hz, left, right, volume)
    })
}

/// Toggles morphable `PeriodicWave` wavetable audition on user button click.
pub fn toggle_studio_wavetable_audition_from_gesture(
    real: &[f32],
    imag: &[f32],
    freq_hz: f32,
    volume: f32,
) -> Result<bool, String> {
    STUDIO_PREVIEW.with(|cell| {
        cell.borrow_mut()
            .toggle_wavetable_audition_from_user_gesture(real, imag, freq_hz, volume)
    })
}

/// Updates morphable `PeriodicWave` partials when audition is already unlocked.
pub fn update_studio_wavetable_audition(
    real: &[f32],
    imag: &[f32],
    freq_hz: f32,
    volume: f32,
) -> Result<(), String> {
    STUDIO_PREVIEW.with(|cell| {
        cell.borrow_mut()
            .update_wavetable_audition(real, imag, freq_hz, volume)
    })
}

/// Suspends studio strike and wavetable audition output cleanly.
pub fn suspend_studio_preview(reason: MuteReason) {
    STUDIO_PREVIEW.with(|cell| cell.borrow_mut().suspend_all(reason));
}

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "studio_preview_wasm_tests.rs"]
mod wasm_tests;
