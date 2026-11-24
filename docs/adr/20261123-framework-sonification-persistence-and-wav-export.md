# ADR 20261123: Framework-Scoped Sonification Persistence, `<= 20 ms` Gain-Ramp Guards, and `ExportQueue` WAV Dispatch

## Status

Accepted (extends [ADR 0040: Dual-Backend Web Audio Sonification](0040-dual-backend-web-audio-sonification.md) and [ADR 20261123: DEFLATE-Compressed Workspace Share URLs and Browser History Bridge](20261123-compressed-workspace-urls-and-history-bridge.md))

## Context

`math_explorer_gui::audio` provides an 8-voice `SonificationFrame` synthesis core backed by `WebAudioNodeBackend` (`web-sys`) on `wasm32-unknown-unknown` and `OfflinePcmSynthesizer` on native targets. However, multi-pane `SimulationFramework` instances, `UnifiedSimTool<M>` workers, JSON parameter presets, `#tab=...&tool=...&state=...` share URLs, `#ws=...` compressed multi-pane workspace URLs, and the unified `ExportQueue` toolbar lacked:

1. A bounded, non-mute `AcousticSettings` (`gain`, `pitch_scale`) contract persisted across tool snapshots, presets, and URL fragments without ever auto-unmuting playback on startup or hydration.
2. Per-framework-instance mute/gesture isolation (`AudioEngineController`), a monotonic `PlatformInstant`-driven `GainRampEnvelope` guaranteeing `<= 20 ms` (`15 ms` target) fade-out to exact `0.0` silence on pause, `is_diverged` numerical divergence (with assertive ARIA live-region alerts), or `webglcontextlost`, and a zero-duplication UI bridge (`ScopedAudioUiBridge`) synchronizing `SimulationFramework`, `PlaybackTransport::with_sonification`, and `render_sonification_panel`.
3. Deterministic `ExportFormat::Wav` (`ExportCapabilities::with_audio()`) action-bar dispatch (`"audio/wav"` MIME type) sampling multi-step trajectories across `SpikeAnalysisTool`, `HodgkinHuxleyTool` (`UnifiedSimTool<HodgkinHuxleyUnified>`), `AttractorPlotter`, `WaveSimulatorTool`, `MriTool`, and `PrimeSpiralWidget`.

## Decision

1. **Persisted Non-Mute `AcousticSettings` and `SonificationFrame::with_pitch_scale` (`math_explorer_gui/src/audio/`)**:
   - Introduce `AcousticSettings` (`gain` in `[0.0, 1.0]`, default `0.5`; `pitch_scale` in `[0.25, 4.0]`, default `1.0`) under `#[serde(deny_unknown_fields)]` with `is_default()`, `validate()`, and `sanitized()`, omitting `muted` so cold boots, preset imports, `#tab=...` URLs, and `#ws=...` workspace URLs always boot muted.
   - Add `SonificationFrame::with_pitch_scale` and `synthesize_wav_16bit` / `write_wav_header`, plus `.wav` -> `"audio/wav"` MIME mapping in `egui_plot::file_io::infer_mime_type`.
2. **Scoped `AudioEngineController`, `GainRampEnvelope`, and `ScopedAudioUiBridge` (`math_explorer_gui/src/audio/` & `math_explorer_gui/src/framework/`)**:
   - Add `soft_clip_sample` / `soft_clip_stereo` (`f32::tanh` soft-clipping) and `GainRampEnvelope` (`15 ms <= 20 ms` via `PlatformInstant::now()`).
   - Embed `AudioEngineController` on `SimulationFramework`, synchronizing per-frame settings and mute gestures via `ScopedAudioUiBridge` across `PlaybackTransport::with_sonification` and `render_sonification_panel`, silencing within `<= 20 ms` on `PlaybackAction::Pause`, `is_diverged` (assertive ARIA announcement), and `webglcontextlost` epoch increments.
3. **Unified `ExportFormat::Wav` & Six Domain Tool Trajectory Hooks**:
   - Extend `InteractiveTool` (`sonification_frame`, `acoustic_settings`, `set_acoustic_settings`, `sample_sonification_frames`), `UnifiedModel` / `UnifiedSimTool<M>`, `ExportCapabilities::with_audio()`, and `ExportFormat::Wav` across `SpikeAnalysisTool`, `HodgkinHuxleyUnified`, `AttractorPlotter` (Lorenz & custom AST 3D flows), `WaveSimulatorTool` (presets, custom `V(x)`, & sculpted barriers), `MriTool` (continuous FID, `InversionPulse`, & 8-voice `HahnSpinEcho`), and `PrimeSpiralWidget`.

## Consequences

- **Positive**: Zero new external crates or native Linux `.deb` audio dependencies; zero-allocation hot-path synthesis verified under `VerifiedAllocator`; lossless `AcousticSettings` persistence across JSON presets, `#tab=...` single-pane URLs, and `#ws=...` compressed multi-pane URLs while maintaining strict user-gesture-gated Web Audio startup; deterministic 16-bit stereo RIFF `.wav` export from `ExportQueue`.
- **Negative / Trade-offs**: Each sonified `InteractiveTool` includes `AcousticSettings` in its `state_revision` hash and `validate()` bounds checks (`<= 4 KiB` single-pane / `<= 32 KiB` workspace URL caps).
