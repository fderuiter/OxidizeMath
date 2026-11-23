# ADR 20261123: PDE Impulse Response and Band-Limited Wavetable `.wav` Studio

## Status

Accepted

## Context

OxidizeMath's `ChladniWaveTankTool` and `Wave2DSolver` simulate Kirchhoff-Love biharmonic thin-plate bending ($u_{tt} + \gamma u_t = -D \nabla^4 u + F$) and 2D hyperbolic acoustic wave propagation ($u_{tt} + \gamma u_t = c(x,y)^2 \nabla^2 u + F$) with Chladni nodal pins, multi-slit barriers, and refractive biconvex lenses, while `audio::wav` previously emitted only 44.1 kHz stereo 16-bit PCM sonification clips. Sound designers, musicians, and acoustics students lacked a way to:

1. Sample off-grid stereo vibrations $(x_L, y_L)$ and $(x_R, y_R)$ after a mallet strike and export studio-grade `44.1 kHz` or `48 kHz`, `PCM16` (`fmt = 1`) or `IEEE Float32` (`fmt = 3`) `.wav` impulse responses for convolution reverbs.
2. Synthesize DC-free, Nyquist band-limited `2048`-sample multi-frame (`1..=256`) wavetable `.wav` banks from symbolic `CompiledExpr` formulas $f(x, m)$ or 2D PDE wave cross-sections with `clm ` and `uhWT` metadata chunks recognized natively by Vital, Serum, and Surge XT.
3. Audition stereo strikes and morphable wavetables in the browser without violating the strict `'self'` Content-Security-Policy (ADR 0040) or browser autoplay user-gesture policies.

## Decision

1. **Zero-Allocation Bilinear PDE Pickup Readout (`pure_math`)**:
   - Add `#[verified_engine::verified] pub fn sample_displacement_bilinear(&self, x_norm: f64, y_norm: f64) -> f64` to `Wave2DSolver` (`crates/pure_math/src/pure_math/analysis/pde/wave_2d.rs`) backed by `sample_bilinear_2d` in `wave_2d/masks.rs` using `flatten_2d_index` and sanitizing non-finite coordinates to `0.0`.
2. **Multi-Format RIFF `WAVE` & Synthesizer Chunk Encoder (`audio::wav`)**:
   - Extend `math_explorer_gui/src/audio/wav.rs` with `WavSampleRate` (`44_100` / `48_000` Hz), `WavSampleFormat` (`PCM16` `fmt = 1` / `IEEE Float32` `fmt = 3`), `WavChannelLayout` (`Mono` / `Stereo`), and optional `WavetableChunkSpec` appending `clm ` (`<!>2048 01000000 OxidizeMath`) and `uhWT` (`[version, frame_count, frame_samples]`) chunks while preserving `encode_wav_stereo_pcm16` and `synthesize_frames_to_wav`.
3. **Pre-Planned `rustfft` Spectral Conditioning (`audio::wavetable`)**:
   - Implement `WavetableSpectralProcessor` with reusable `2048`-point forward/inverse `rustfft` plans and pre-allocated scratch buffers (`process_with_scratch`) to zero DC offset (`< 1e-6`), brickwall band-limit harmonics above Nyquist, peak-normalize headroom, and extract `PeriodicWave` cosine/sine partials with zero heap allocations inside inner loops.
4. **Cooperative Stereo IR Rendering in `ChladniWaveTankTool`**:
   - Add draggable canvas stereo pickups `(x_L, y_L)` and `(x_R, y_R)`, strike audition, and cooperative `0.5 s`–`4.0 s` IR rendering sliced via `PlatformInstant::now` to `<= 6.0 ms` per UI frame so WASM frame pacing stays smoothly below `8.0 ms`.
5. **`WavetableStudioTool` & Gesture-Gated `StudioPreviewController`**:
   - Register `WavetableStudioTool` (`tabs/analysis/wavetable_studio/`) combining `MathExpressionInputBar` (`f(x, m)`) and `Wave2DSolver` cross-section capture with dual `egui_plot` waveform/partial views.
   - Add `audio/studio_preview.rs` beside `web_audio.rs` providing `AudioBufferSourceNode` stereo strike playback and morphable `OscillatorNode` + `PeriodicWave` audition routed through a `-3.0 dBFS` `DynamicsCompressorNode`, unlocked strictly inside user button gestures with zero `AudioWorklet` scripts.

## Consequences

- Exported stereo plate IRs match Leissa's analytical clamped-plate mode ratios (`analytic_clamped_square_omega`) within `2.5%` and reflect multi-slit diffraction and biconvex refractive lens masks.
- Exported `2048 * frame_count` wavetable `.wav` banks load directly into Vital, Serum, and Surge XT with zero DC offset (`< 1e-6`) and zero aliasing partials above Nyquist.
- All inner PDE pickup and `2048`-sample wavetable synthesis loops execute with zero heap allocations under `VerifiedEngine::run_verified`, preserving strict NASA Power of 10 and WASM bundle budget guarantees.
