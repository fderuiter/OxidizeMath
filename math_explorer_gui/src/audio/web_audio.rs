//! Native browser `WebAudioNodeBackend` (`wasm32-unknown-unknown`) without `AudioWorklet` scripts.

use wasm_bindgen::JsValue;
use web_sys::{
    AnalyserNode, AudioContext, AudioContextState, AudioDestinationNode, BiquadFilterNode,
    BiquadFilterType, DynamicsCompressorNode, GainNode, OscillatorNode, OscillatorType,
    PeriodicWave, PeriodicWaveOptions, StereoPannerNode,
};

use super::backend::{MuteReason, SonificationBackend};
use super::fourier::PresetHarmonicBank;
use super::frame::{
    clamp_freq_hz, sanitize_f32, HarmonicWeights, MeterReading, SonificationFrame, VoiceSpec,
    WaveformPreset, LIMITER_CEILING_DBFS, LIMITER_CEILING_LINEAR, MAX_VOICES, MUTE_RAMP_TAU_SEC,
    NUM_HARMONICS, PARAM_SMOOTH_TAU_SEC,
};

struct WebVoiceNodes {
    carrier: OscillatorNode,
    mod_osc: OscillatorNode,
    mod_gain: GainNode,
    filter: BiquadFilterNode,
    voice_gain: GainNode,
    panner: StereoPannerNode,
    prev_impulse: f32,
}

struct WebAudioGraph {
    ctx: AudioContext,
    voices: Vec<WebVoiceNodes>,
    preset_waves: Vec<PeriodicWave>,
    noise_filter: BiquadFilterNode,
    noise_gain: GainNode,
    master_gain: GainNode,
    _compressor: DynamicsCompressorNode,
    _ceiling_gain: GainNode,
    analyser: AnalyserNode,
    _destination: AudioDestinationNode,
    active_preset: WaveformPreset,
    using_custom_wave: bool,
}

/// Browser Web Audio mixer node backend unlocked strictly on explicit user gesture.
pub struct WebAudioNodeBackend {
    graph: Option<WebAudioGraph>,
    preset_bank: PresetHarmonicBank,
    current_frame: SonificationFrame,
    master_volume: f32,
    muted: bool,
    last_mute_reason: MuteReason,
    last_meter: MeterReading,
}

impl Default for WebAudioNodeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl WebAudioNodeBackend {
    /// Creates a muted backend with `graph: None`; no `AudioContext` is constructed until user gesture.
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: None,
            preset_bank: PresetHarmonicBank::new(),
            current_frame: SonificationFrame::silent(),
            master_volume: 0.80,
            muted: true,
            last_mute_reason: MuteReason::InitialDefault,
            last_meter: MeterReading::silent(),
        }
    }

    /// Returns `true` only after an explicit user gesture has instantiated an `AudioContext`.
    #[must_use]
    pub fn has_audio_context(&self) -> bool {
        self.graph.is_some()
    }

    /// Returns the underlying `AudioContextState` when initialized.
    #[must_use]
    pub fn context_state(&self) -> Option<AudioContextState> {
        self.graph.as_ref().map(|g| g.ctx.state())
    }

    /// Last mute lifecycle reason recorded by the backend.
    #[must_use]
    pub const fn last_mute_reason(&self) -> MuteReason {
        self.last_mute_reason
    }

    /// Number of spectrum frequency bins configured on the post-limiter `AnalyserNode`.
    #[must_use]
    pub fn analyser_bin_count(&self) -> Option<u32> {
        self.graph
            .as_ref()
            .map(|g| g.analyser.frequency_bin_count())
    }
}

impl SonificationBackend for WebAudioNodeBackend {
    fn is_muted(&self) -> bool {
        self.muted
    }

    fn set_muted_from_user_gesture(&mut self, muted: bool) -> Result<(), String> {
        if muted {
            self.mute_and_suspend(MuteReason::UserMuted);
            return Ok(());
        }
        if self.graph.is_none() {
            self.graph = Some(build_audio_graph(&self.preset_bank)?);
        }
        self.muted = false;
        if let Some(graph) = &mut self.graph {
            if graph.ctx.state() == AudioContextState::Suspended {
                let _ = graph.ctx.resume().map_err(format_js_err)?;
            }
            apply_frame_to_graph(graph, &self.current_frame, self.master_volume)?;
        }
        self.last_meter = self.current_frame.estimate_meter(self.master_volume, false);
        Ok(())
    }

    fn mute_and_suspend(&mut self, reason: MuteReason) {
        self.muted = true;
        self.last_mute_reason = reason;
        self.last_meter = MeterReading::silent();
        if let Some(graph) = &self.graph {
            let now = graph.ctx.current_time();
            let _ = graph.master_gain.gain().cancel_scheduled_values(now);
            let _ = graph
                .master_gain
                .gain()
                .set_target_at_time(0.0, now, MUTE_RAMP_TAU_SEC);
            let _ = graph
                .noise_gain
                .gain()
                .set_target_at_time(0.0, now, MUTE_RAMP_TAU_SEC);
            for voice in &graph.voices {
                let _ = voice
                    .voice_gain
                    .gain()
                    .set_target_at_time(0.0, now, MUTE_RAMP_TAU_SEC);
            }
            if graph.ctx.state() == AudioContextState::Running {
                let _ = graph.ctx.suspend();
            }
        }
    }

    fn apply_frame(&mut self, frame: &SonificationFrame) {
        let clean = frame.sanitize();
        self.current_frame = clean;
        self.last_meter = clean.estimate_meter(self.master_volume, self.muted);
        if !self.muted {
            if let Some(graph) = &mut self.graph {
                let _ = apply_frame_to_graph(graph, &clean, self.master_volume);
            }
        }
    }

    fn set_master_volume(&mut self, volume: f32) {
        self.master_volume = sanitize_f32(volume, 0.80, 0.0, 1.0);
        self.last_meter = self
            .current_frame
            .estimate_meter(self.master_volume, self.muted);
        if !self.muted {
            if let Some(graph) = &self.graph {
                let now = graph.ctx.current_time();
                let target = self.current_frame.master_gain * self.master_volume;
                let _ =
                    graph
                        .master_gain
                        .gain()
                        .set_target_at_time(target, now, PARAM_SMOOTH_TAU_SEC);
            }
        }
    }

    fn master_volume(&self) -> f32 {
        self.master_volume
    }

    fn meter_reading(&self) -> MeterReading {
        self.last_meter
    }
}

fn build_audio_graph(preset_bank: &PresetHarmonicBank) -> Result<WebAudioGraph, String> {
    let ctx = AudioContext::new().map_err(format_js_err)?;
    let master_gain = ctx.create_gain().map_err(format_js_err)?;
    master_gain.gain().set_value(0.0);

    let compressor = build_master_limiter(&ctx)?;
    let ceiling_gain = ctx.create_gain().map_err(format_js_err)?;
    ceiling_gain.gain().set_value(LIMITER_CEILING_LINEAR);

    let analyser = ctx.create_analyser().map_err(format_js_err)?;
    analyser.set_fft_size(256);
    let destination = ctx.destination();

    master_gain
        .connect_with_audio_node(&compressor)
        .map_err(format_js_err)?;
    compressor
        .connect_with_audio_node(&ceiling_gain)
        .map_err(format_js_err)?;
    ceiling_gain
        .connect_with_audio_node(&analyser)
        .map_err(format_js_err)?;
    analyser
        .connect_with_audio_node(&destination)
        .map_err(format_js_err)?;

    let mut preset_waves = Vec::with_capacity(WaveformPreset::ALL.len());
    for preset in WaveformPreset::ALL {
        preset_waves.push(create_periodic_wave(&ctx, &preset_bank.get(preset))?);
    }

    let mut voices = Vec::with_capacity(MAX_VOICES);
    for _ in 0..MAX_VOICES {
        voices.push(build_voice_strip(&ctx, &master_gain, &preset_waves[0])?);
    }
    let (noise_filter, noise_gain) = build_chaotic_noise_strip(&ctx, &master_gain)?;

    Ok(WebAudioGraph {
        ctx,
        voices,
        preset_waves,
        noise_filter,
        noise_gain,
        master_gain,
        _compressor: compressor,
        _ceiling_gain: ceiling_gain,
        analyser,
        _destination: destination,
        active_preset: WaveformPreset::Sine,
        using_custom_wave: false,
    })
}

fn build_master_limiter(ctx: &AudioContext) -> Result<DynamicsCompressorNode, String> {
    let comp = ctx.create_dynamics_compressor().map_err(format_js_err)?;
    comp.threshold().set_value(LIMITER_CEILING_DBFS);
    comp.knee().set_value(0.0);
    comp.ratio().set_value(20.0);
    comp.attack().set_value(0.001);
    comp.release().set_value(0.050);
    Ok(comp)
}

fn build_voice_strip(
    ctx: &AudioContext,
    master_gain: &GainNode,
    initial_wave: &PeriodicWave,
) -> Result<WebVoiceNodes, String> {
    let carrier = ctx.create_oscillator().map_err(format_js_err)?;
    carrier.set_type(OscillatorType::Sine);
    carrier.set_periodic_wave(initial_wave);
    carrier.frequency().set_value(220.0);

    let mod_osc = ctx.create_oscillator().map_err(format_js_err)?;
    mod_osc.set_type(OscillatorType::Sine);
    mod_osc.frequency().set_value(110.0);

    let mod_gain = ctx.create_gain().map_err(format_js_err)?;
    mod_gain.gain().set_value(0.0);
    mod_osc
        .connect_with_audio_node(&mod_gain)
        .map_err(format_js_err)?;
    mod_gain
        .connect_with_audio_param(&carrier.frequency())
        .map_err(format_js_err)?;

    let filter = ctx.create_biquad_filter().map_err(format_js_err)?;
    filter.set_type(BiquadFilterType::Lowpass);
    filter.frequency().set_value(3200.0);

    let voice_gain = ctx.create_gain().map_err(format_js_err)?;
    voice_gain.gain().set_value(0.0);

    let panner = ctx.create_stereo_panner().map_err(format_js_err)?;
    panner.pan().set_value(0.0);

    carrier
        .connect_with_audio_node(&filter)
        .map_err(format_js_err)?;
    filter
        .connect_with_audio_node(&voice_gain)
        .map_err(format_js_err)?;
    voice_gain
        .connect_with_audio_node(&panner)
        .map_err(format_js_err)?;
    panner
        .connect_with_audio_node(master_gain)
        .map_err(format_js_err)?;

    carrier.start().map_err(format_js_err)?;
    mod_osc.start().map_err(format_js_err)?;

    Ok(WebVoiceNodes {
        carrier,
        mod_osc,
        mod_gain,
        filter,
        voice_gain,
        panner,
        prev_impulse: 0.0,
    })
}

fn build_chaotic_noise_strip(
    ctx: &AudioContext,
    master_gain: &GainNode,
) -> Result<(BiquadFilterNode, GainNode), String> {
    let osc_a = ctx.create_oscillator().map_err(format_js_err)?;
    osc_a.set_type(OscillatorType::Sawtooth);
    osc_a.frequency().set_value(587.33);

    let osc_b = ctx.create_oscillator().map_err(format_js_err)?;
    osc_b.set_type(OscillatorType::Square);
    osc_b.frequency().set_value(950.32);

    let fm_gain = ctx.create_gain().map_err(format_js_err)?;
    fm_gain.gain().set_value(4200.0);
    osc_b
        .connect_with_audio_node(&fm_gain)
        .map_err(format_js_err)?;
    fm_gain
        .connect_with_audio_param(&osc_a.frequency())
        .map_err(format_js_err)?;

    let filter = ctx.create_biquad_filter().map_err(format_js_err)?;
    filter.set_type(BiquadFilterType::Bandpass);
    filter.frequency().set_value(1400.0);
    filter.q().set_value(0.7);

    let gain = ctx.create_gain().map_err(format_js_err)?;
    gain.gain().set_value(0.0);

    osc_a
        .connect_with_audio_node(&filter)
        .map_err(format_js_err)?;
    filter
        .connect_with_audio_node(&gain)
        .map_err(format_js_err)?;
    gain.connect_with_audio_node(master_gain)
        .map_err(format_js_err)?;

    osc_a.start().map_err(format_js_err)?;
    osc_b.start().map_err(format_js_err)?;
    Ok((filter, gain))
}

/// Builds a Web Audio `PeriodicWave` via `PeriodicWaveOptions` from Fourier harmonic weights.
pub fn create_periodic_wave(
    ctx: &AudioContext,
    harmonics: &HarmonicWeights,
) -> Result<PeriodicWave, String> {
    let mut real = [0.0_f32; NUM_HARMONICS + 1];
    let mut imag = [0.0_f32; NUM_HARMONICS + 1];
    real[1..].copy_from_slice(&harmonics.cos_a);
    imag[1..].copy_from_slice(&harmonics.sin_b);

    let real_arr = js_sys::Float32Array::from(&real[..]);
    let imag_arr = js_sys::Float32Array::from(&imag[..]);
    let opts = PeriodicWaveOptions::new();
    opts.set_real(real_arr.as_ref());
    opts.set_imag(imag_arr.as_ref());
    PeriodicWave::new_with_options(ctx, &opts).map_err(format_js_err)
}

fn apply_frame_to_graph(
    graph: &mut WebAudioGraph,
    frame: &SonificationFrame,
    master_volume: f32,
) -> Result<(), String> {
    let now = graph.ctx.current_time();
    if let Some(custom) = &frame.custom_harmonics {
        let wave = create_periodic_wave(&graph.ctx, custom)?;
        for v in &graph.voices {
            v.carrier.set_periodic_wave(&wave);
        }
        graph.using_custom_wave = true;
    } else if graph.using_custom_wave || graph.active_preset != frame.waveform {
        let wave = &graph.preset_waves[frame.waveform.index()];
        for v in &graph.voices {
            v.carrier.set_periodic_wave(wave);
        }
        graph.active_preset = frame.waveform;
        graph.using_custom_wave = false;
    }

    let active = frame.active_voices as usize;
    for (idx, nodes) in graph.voices.iter_mut().enumerate() {
        let spec = if idx < active {
            frame.voices[idx]
        } else {
            VoiceSpec::silent()
        };
        update_voice_params(nodes, &spec, now)?;
    }

    graph
        .noise_filter
        .frequency()
        .set_target_at_time(
            clamp_freq_hz(frame.noise_cutoff_hz),
            now,
            PARAM_SMOOTH_TAU_SEC,
        )
        .map_err(format_js_err)?;
    graph
        .noise_gain
        .gain()
        .set_target_at_time(frame.noise_gain * 0.25, now, PARAM_SMOOTH_TAU_SEC)
        .map_err(format_js_err)?;
    graph
        .master_gain
        .gain()
        .set_target_at_time(frame.master_gain * master_volume, now, PARAM_SMOOTH_TAU_SEC)
        .map_err(format_js_err)?;
    Ok(())
}

fn update_voice_params(
    nodes: &mut WebVoiceNodes,
    spec: &VoiceSpec,
    now: f64,
) -> Result<(), String> {
    let freq = clamp_freq_hz(spec.freq_hz);
    let mod_hz = spec.fm_mod_hz.clamp(0.0, 4000.0);
    let dev_hz = (spec.fm_index * mod_hz).clamp(0.0, 3000.0);
    nodes
        .carrier
        .frequency()
        .set_target_at_time(freq, now, PARAM_SMOOTH_TAU_SEC)
        .map_err(format_js_err)?;
    nodes
        .mod_osc
        .frequency()
        .set_target_at_time(mod_hz.max(0.5), now, PARAM_SMOOTH_TAU_SEC)
        .map_err(format_js_err)?;
    nodes
        .mod_gain
        .gain()
        .set_target_at_time(dev_hz, now, PARAM_SMOOTH_TAU_SEC)
        .map_err(format_js_err)?;
    nodes
        .filter
        .frequency()
        .set_target_at_time(
            clamp_freq_hz(spec.filter_cutoff_hz),
            now,
            PARAM_SMOOTH_TAU_SEC,
        )
        .map_err(format_js_err)?;
    nodes
        .panner
        .pan()
        .set_target_at_time(spec.pan.clamp(-1.0, 1.0), now, PARAM_SMOOTH_TAU_SEC)
        .map_err(format_js_err)?;

    let triggered = spec.transient_impulse > 0.25 && nodes.prev_impulse <= 0.25;
    nodes.prev_impulse = spec.transient_impulse;
    if triggered {
        let burst = (spec.gain + 0.35 * spec.transient_impulse).clamp(0.0, 1.0);
        nodes
            .voice_gain
            .gain()
            .set_target_at_time(burst, now, 0.002)
            .map_err(format_js_err)?;
        nodes
            .voice_gain
            .gain()
            .set_target_at_time(spec.gain, now + 0.006, PARAM_SMOOTH_TAU_SEC)
            .map_err(format_js_err)?;
    } else {
        nodes
            .voice_gain
            .gain()
            .set_target_at_time(spec.gain, now, PARAM_SMOOTH_TAU_SEC)
            .map_err(format_js_err)?;
    }
    Ok(())
}

fn format_js_err(err: JsValue) -> String {
    err.as_string()
        .unwrap_or_else(|| format!("Web Audio error: {err:?}"))
}

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "web_audio_wasm_tests.rs"]
mod wasm_tests;
