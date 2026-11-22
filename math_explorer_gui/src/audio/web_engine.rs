//! W3C Web Audio API 1.1 live browser synthesis graph via `web-sys` (`wasm32` only).
//!
//! Lazily instantiates `AudioContext` strictly after an explicit user gesture, routes voices
//! through `OscillatorNode`, `PeriodicWave`, `BiquadFilterNode`, `StereoPannerNode`,
//! `AudioBuffer` + `AudioBufferSourceNode`, `DynamicsCompressorNode`, and `f32::tanh`-limited
//! `GainNode`s, and ramps gain to zero within `15 ms` (`<= 20 ms`) on pause, divergence, or
//! `webglcontextlost`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::frame::{
    AcousticSettings, SonificationFrame, MAX_HARMONIC_SLOTS, MAX_POLYRHYTHM_VOICES,
};
use super::limiter::{soft_clip_sample, SILENCE_RAMP_DURATION_SECS};
use crate::web_io::PlatformInstant;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;
use std::cell::RefCell;
use std::rc::{Rc, Weak};
use web_sys::{
    AudioBuffer, AudioBufferSourceNode, AudioContext, AudioContextState, BiquadFilterNode,
    BiquadFilterType, DynamicsCompressorNode, GainNode, OscillatorNode, OscillatorType,
    PeriodicWave, PeriodicWaveOptions, StereoPannerNode,
};

const CLICK_BUFFER_SAMPLES: u32 = 512;
const CLICK_COOLDOWN_MS: u64 = 25;

struct SharedWebContextHandle {
    ctx: AudioContext,
    master_gain: GainNode,
}

thread_local! {
    static ACTIVE_WEB_CONTEXTS: RefCell<Vec<Weak<SharedWebContextHandle>>> =
        const { RefCell::new(Vec::new()) };
}

fn ramp_gain_node_to_zero(ctx: &AudioContext, gain: &GainNode) {
    let now = ctx.current_time();
    let param = gain.gain();
    let current = param.value();
    let _ = param.cancel_scheduled_values(now);
    let _ = param.set_value_at_time(current, now);
    let _ = param.linear_ramp_to_value_at_time(0.0, now + f64::from(SILENCE_RAMP_DURATION_SECS));
}

/// Immediately ramps all active Web Audio master gains to `0.0` within `15 ms` (`<= 20 ms`)
/// and suspends active `AudioContext`s when `webglcontextlost` fires.
pub fn silence_all_web_audio_contexts() -> usize {
    let mut silenced = 0_usize;
    ACTIVE_WEB_CONTEXTS.with(|list| {
        let mut vec = list.borrow_mut();
        vec.retain(|weak| {
            if let Some(rc) = weak.upgrade() {
                ramp_gain_node_to_zero(&rc.ctx, &rc.master_gain);
                let _ = rc.ctx.suspend();
                silenced += 1;
                true
            } else {
                false
            }
        });
    });
    silenced
}

struct WebAudioNodes {
    shared: Rc<SharedWebContextHandle>,
    carrier: OscillatorNode,
    modulator: OscillatorNode,
    fm_depth: GainNode,
    poly_oscs: [OscillatorNode; MAX_POLYRHYTHM_VOICES],
    poly_gains: [GainNode; MAX_POLYRHYTHM_VOICES],
    filter: BiquadFilterNode,
    panner: StereoPannerNode,
    voice_gain: GainNode,
    compressor: DynamicsCompressorNode,
    click_buffer: AudioBuffer,
    using_periodic_wave: bool,
    last_harmonics_real: [f32; MAX_HARMONIC_SLOTS],
    last_harmonics_imag: [f32; MAX_HARMONIC_SLOTS],
    last_click_at: Option<PlatformInstant>,
}

/// Browser W3C Web Audio API 1.1 synthesizer engine.
#[derive(Default)]
pub struct WebAudioEngine {
    nodes: Option<WebAudioNodes>,
}

impl WebAudioEngine {
    /// Returns `true` if a live `AudioContext` has been created following a user gesture.
    #[must_use]
    pub fn has_audio_context(&self) -> bool {
        self.nodes.is_some()
    }

    fn build_click_buffer(ctx: &AudioContext) -> Result<AudioBuffer, wasm_bindgen::JsValue> {
        let sr = ctx.sample_rate().max(8_000.0);
        let buffer = ctx.create_buffer(1, CLICK_BUFFER_SAMPLES, sr)?;
        let mut samples = [0.0_f32; CLICK_BUFFER_SAMPLES as usize];
        let mut rng = OxidizeRng::new(0x5041_5443_485F_4848);
        for (i, out) in samples.iter_mut().enumerate() {
            let env = (-6.0 * (i as f32) / (CLICK_BUFFER_SAMPLES as f32)).exp();
            let noise = ((rng.next_u32() as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
            *out = soft_clip_sample(noise * env * 0.85);
        }
        buffer.copy_to_channel(&samples, 0)?;
        Ok(buffer)
    }

    fn configure_compressor(comp: &DynamicsCompressorNode, now: f64) {
        let _ = comp.threshold().set_value_at_time(-14.0, now);
        let _ = comp.knee().set_value_at_time(12.0, now);
        let _ = comp.ratio().set_value_at_time(8.0, now);
        let _ = comp.attack().set_value_at_time(0.002, now);
        let _ = comp.release().set_value_at_time(0.080, now);
    }

    fn build_poly_voices(
        ctx: &AudioContext,
        filter: &BiquadFilterNode,
        now: f64,
    ) -> Result<
        (
            [OscillatorNode; MAX_POLYRHYTHM_VOICES],
            [GainNode; MAX_POLYRHYTHM_VOICES],
        ),
        wasm_bindgen::JsValue,
    > {
        let mut poly_oscs_vec = Vec::with_capacity(MAX_POLYRHYTHM_VOICES);
        let mut poly_gains_vec = Vec::with_capacity(MAX_POLYRHYTHM_VOICES);
        for _ in 0..MAX_POLYRHYTHM_VOICES {
            let osc = ctx.create_oscillator()?;
            osc.set_type(OscillatorType::Triangle);
            let g = ctx.create_gain()?;
            let _ = g.gain().set_value_at_time(0.0, now);
            osc.connect_with_audio_node(&g)?;
            g.connect_with_audio_node(filter)?;
            osc.start()?;
            poly_oscs_vec.push(osc);
            poly_gains_vec.push(g);
        }
        let Ok(poly_oscs) = poly_oscs_vec.try_into() else {
            return Err(wasm_bindgen::JsValue::from_str("poly_oscs length mismatch"));
        };
        let Ok(poly_gains) = poly_gains_vec.try_into() else {
            return Err(wasm_bindgen::JsValue::from_str(
                "poly_gains length mismatch",
            ));
        };
        Ok((poly_oscs, poly_gains))
    }

    fn init_graph() -> Result<WebAudioNodes, wasm_bindgen::JsValue> {
        let ctx = AudioContext::new()?;
        let now = ctx.current_time();
        let master_gain = ctx.create_gain()?;
        let _ = master_gain.gain().set_value_at_time(0.0, now);

        let compressor = ctx.create_dynamics_compressor()?;
        Self::configure_compressor(&compressor, now);

        let voice_gain = ctx.create_gain()?;
        let _ = voice_gain.gain().set_value_at_time(0.0, now);
        let panner = ctx.create_stereo_panner()?;
        let filter = ctx.create_biquad_filter()?;
        filter.set_type(BiquadFilterType::Lowpass);

        let carrier = ctx.create_oscillator()?;
        carrier.set_type(OscillatorType::Sine);
        let modulator = ctx.create_oscillator()?;
        modulator.set_type(OscillatorType::Sine);
        let fm_depth = ctx.create_gain()?;
        let _ = fm_depth.gain().set_value_at_time(0.0, now);

        modulator.connect_with_audio_node(&fm_depth)?;
        fm_depth.connect_with_audio_param(&carrier.frequency())?;
        carrier.connect_with_audio_node(&filter)?;

        let (poly_oscs, poly_gains) = Self::build_poly_voices(&ctx, &filter, now)?;

        filter.connect_with_audio_node(&panner)?;
        panner.connect_with_audio_node(&voice_gain)?;
        voice_gain.connect_with_audio_node(&compressor)?;
        compressor.connect_with_audio_node(&master_gain)?;
        master_gain.connect_with_audio_node(&ctx.destination())?;

        carrier.start()?;
        modulator.start()?;
        let click_buffer = Self::build_click_buffer(&ctx)?;
        let shared = Rc::new(SharedWebContextHandle { ctx, master_gain });
        ACTIVE_WEB_CONTEXTS.with(|list| list.borrow_mut().push(Rc::downgrade(&shared)));

        Ok(WebAudioNodes {
            shared,
            carrier,
            modulator,
            fm_depth,
            poly_oscs,
            poly_gains,
            filter,
            panner,
            voice_gain,
            compressor,
            click_buffer,
            using_periodic_wave: false,
            last_harmonics_real: [0.0; MAX_HARMONIC_SLOTS],
            last_harmonics_imag: [0.0; MAX_HARMONIC_SLOTS],
            last_click_at: None,
        })
    }

    /// Ensures the `AudioContext` is initialized ONLY when `has_user_gesture` is true.
    pub fn ensure_started(&mut self, has_user_gesture: bool) -> bool {
        if !has_user_gesture {
            return false;
        }
        if self.nodes.is_none() {
            self.nodes = Self::init_graph().ok();
        }
        if let Some(ref nodes) = self.nodes {
            if nodes.shared.ctx.state() == AudioContextState::Suspended {
                let _ = nodes.shared.ctx.resume();
            }
            let _ = &nodes.compressor;
            true
        } else {
            false
        }
    }

    /// Ramps master and voice gains to `0.0` within `15 ms` (`<= 20 ms`).
    pub fn ramp_to_silence(&mut self) {
        if let Some(ref nodes) = self.nodes {
            ramp_gain_node_to_zero(&nodes.shared.ctx, &nodes.shared.master_gain);
            ramp_gain_node_to_zero(&nodes.shared.ctx, &nodes.voice_gain);
            for g in &nodes.poly_gains {
                ramp_gain_node_to_zero(&nodes.shared.ctx, g);
            }
        }
    }

    fn apply_periodic_wave_if_needed(nodes: &mut WebAudioNodes, f: &SonificationFrame) {
        if f.use_periodic_wave {
            let changed = !nodes.using_periodic_wave
                || nodes.last_harmonics_real != f.harmonics_real
                || nodes.last_harmonics_imag != f.harmonics_imag;
            if changed {
                let opts = PeriodicWaveOptions::new();
                opts.set_real(&js_sys::Float32Array::from(&f.harmonics_real[..]));
                opts.set_imag(&js_sys::Float32Array::from(&f.harmonics_imag[..]));
                if let Ok(wave) = PeriodicWave::new_with_options(&nodes.shared.ctx, &opts) {
                    nodes.carrier.set_periodic_wave(&wave);
                    nodes.using_periodic_wave = true;
                    nodes.last_harmonics_real = f.harmonics_real;
                    nodes.last_harmonics_imag = f.harmonics_imag;
                }
            }
        } else if nodes.using_periodic_wave {
            nodes.carrier.set_type(OscillatorType::Sine);
            nodes.using_periodic_wave = false;
        }
    }

    fn maybe_trigger_click(nodes: &mut WebAudioNodes, impulse: f32, master_gain: f32) {
        if impulse <= 0.04 || master_gain <= 0.001 {
            return;
        }
        let now_inst = PlatformInstant::now();
        if let Some(prev) = nodes.last_click_at {
            if now_inst.duration_since(prev).as_millis() < u128::from(CLICK_COOLDOWN_MS) {
                return;
            }
        }
        nodes.last_click_at = Some(now_inst);
        if let Ok(src) = AudioBufferSourceNode::new(&nodes.shared.ctx) {
            src.set_buffer(Some(&nodes.click_buffer));
            if let Ok(click_gain) = nodes.shared.ctx.create_gain() {
                let g = soft_clip_sample(impulse * 0.7);
                let _ = click_gain
                    .gain()
                    .set_value_at_time(g, nodes.shared.ctx.current_time());
                if src.connect_with_audio_node(&click_gain).is_ok()
                    && click_gain.connect_with_audio_node(&nodes.filter).is_ok()
                {
                    let _ = src.start();
                }
            }
        }
    }

    /// Updates browser Web Audio node parameters for the current `SonificationFrame`.
    pub fn apply_frame(
        &mut self,
        frame: SonificationFrame,
        settings: AcousticSettings,
        muted: bool,
        has_user_gesture: bool,
    ) {
        if muted || !has_user_gesture || !self.ensure_started(has_user_gesture) {
            self.ramp_to_silence();
            return;
        }
        let f = frame.sanitized();
        if !f.active {
            self.ramp_to_silence();
            return;
        }
        let cfg = settings.sanitized();
        if let Some(ref mut nodes) = self.nodes {
            Self::update_active_nodes(nodes, &f, cfg);
        }
    }

    fn update_active_nodes(
        nodes: &mut WebAudioNodes,
        f: &SonificationFrame,
        cfg: AcousticSettings,
    ) {
        let ramp_end = nodes.shared.ctx.current_time() + f64::from(SILENCE_RAMP_DURATION_SECS);
        Self::apply_periodic_wave_if_needed(nodes, f);

        let carrier_hz = (f.carrier_hz * cfg.pitch_scale).clamp(20.0, 12_000.0);
        let mod_hz = (f.fm_modulator_hz * cfg.pitch_scale).clamp(0.0, 8_000.0);
        let _ = nodes
            .carrier
            .frequency()
            .linear_ramp_to_value_at_time(carrier_hz, ramp_end);
        let _ = nodes
            .modulator
            .frequency()
            .linear_ramp_to_value_at_time(mod_hz, ramp_end);
        let _ = nodes
            .fm_depth
            .gain()
            .linear_ramp_to_value_at_time(f.fm_index_hz, ramp_end);

        nodes.filter.set_type(if f.filter_bandpass {
            BiquadFilterType::Bandpass
        } else {
            BiquadFilterType::Lowpass
        });
        let _ = nodes
            .filter
            .frequency()
            .linear_ramp_to_value_at_time(f.filter_cutoff_hz, ramp_end);
        let _ = nodes
            .filter
            .q()
            .linear_ramp_to_value_at_time(f.filter_q, ramp_end);
        let _ = nodes
            .panner
            .pan()
            .linear_ramp_to_value_at_time(f.pan, ramp_end);

        let voice_amp = soft_clip_sample(f.amplitude * 0.65);
        let master_amp = soft_clip_sample(cfg.gain);
        let _ = nodes
            .voice_gain
            .gain()
            .linear_ramp_to_value_at_time(voice_amp, ramp_end);
        let _ = nodes
            .shared
            .master_gain
            .gain()
            .linear_ramp_to_value_at_time(master_amp, ramp_end);

        for i in 0..MAX_POLYRHYTHM_VOICES {
            let hz = (f.poly_voices_hz[i] * cfg.pitch_scale).clamp(20.0, 12_000.0);
            let amp = soft_clip_sample(f.poly_voices_amp[i] * 0.35);
            let _ = nodes.poly_oscs[i]
                .frequency()
                .linear_ramp_to_value_at_time(hz, ramp_end);
            let _ = nodes.poly_gains[i]
                .gain()
                .linear_ramp_to_value_at_time(amp, ramp_end);
        }
        Self::maybe_trigger_click(nodes, f.click_impulse, master_amp);
    }
}

impl Drop for WebAudioEngine {
    fn drop(&mut self) {
        self.ramp_to_silence();
        if let Some(nodes) = self.nodes.take() {
            let _ = nodes.shared.ctx.close();
        }
    }
}

#[cfg(test)]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn test_web_audio_engine_boots_without_context_until_user_gesture() {
        let mut engine = WebAudioEngine::default();
        assert!(!engine.has_audio_context());
        assert!(!engine.ensure_started(false));
        assert!(!engine.has_audio_context());

        engine.apply_frame(
            SonificationFrame {
                active: true,
                amplitude: 0.6,
                ..SonificationFrame::silent()
            },
            AcousticSettings::default(),
            true,
            false,
        );
        assert!(!engine.has_audio_context());
        let _ = silence_all_web_audio_contexts();
    }
}
