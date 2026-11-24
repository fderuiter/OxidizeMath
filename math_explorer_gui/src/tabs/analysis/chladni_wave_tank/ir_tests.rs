#![cfg_attr(any(), verified(opt_out = "tests"))]

// [cite:graph_parameters_rust]

use eframe::egui::{vec2, Context, Pos2, RawInput, Rect};
use math_explorer::pure_math::analysis::pde::wave_2d::{
    analytic_clamped_square_omega, BoundaryMode2D, Wave2DMedium,
};
use num_complex::Complex;
use rustfft::FftPlanner;
use verified_engine::engine::VerifiedEngine;

use super::*;
use crate::audio::studio_preview::snapshot_studio_preview;
use crate::audio::wav::{inspect_wav_riff, WavSampleFormat, WavSampleRate};

fn decode_float32_stereo_wav(wav: &[u8]) -> (u32, Vec<f32>, Vec<f32>) {
    let summary = inspect_wav_riff(wav).expect("valid Float32 stereo WAV");
    assert_eq!(summary.format_tag, 3);
    assert_eq!(summary.channels, 2);
    let frames = (summary.data_bytes as usize) / 8;
    let mut left = Vec::with_capacity(frames);
    let mut right = Vec::with_capacity(frames);
    let start = summary.data_offset;
    for i in 0..frames {
        let off = start + i * 8;
        left.push(f32::from_le_bytes(wav[off..off + 4].try_into().unwrap()));
        right.push(f32::from_le_bytes(
            wav[off + 4..off + 8].try_into().unwrap(),
        ));
    }
    (summary.sample_rate_hz, left, right)
}

fn find_parabolic_fft_peak_hz(
    signal: &[f32],
    sample_rate_hz: u32,
    search_min_hz: f64,
    search_max_hz: f64,
) -> f64 {
    let n = signal.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buf: Vec<Complex<f64>> = signal
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let w = 0.5 * (1.0 - (std::f64::consts::TAU * (i as f64) / (n as f64)).cos());
            Complex::new((s as f64) * w, 0.0)
        })
        .collect();
    fft.process(&mut buf);

    let df = (sample_rate_hz as f64) / (n as f64);
    let bin_lo = ((search_min_hz / df).floor() as usize).clamp(2, n / 2 - 3);
    let bin_hi = ((search_max_hz / df).ceil() as usize).clamp(bin_lo + 1, n / 2 - 2);

    let mut best_bin = bin_lo;
    let mut best_mag = -1.0_f64;
    for (k, slot) in buf.iter().enumerate().take(bin_hi + 1).skip(bin_lo) {
        let m = slot.norm();
        if m > best_mag {
            best_mag = m;
            best_bin = k;
        }
    }
    let m_prev = buf[best_bin - 1].norm();
    let m_curr = buf[best_bin].norm();
    let m_next = buf[best_bin + 1].norm();
    let denom = m_prev - 2.0 * m_curr + m_next;
    let delta = if denom.abs() > 1.0e-12 {
        0.5 * (m_prev - m_next) / denom
    } else {
        0.0
    };
    ((best_bin as f64) + delta.clamp(-0.5, 0.5)) * df
}

#[test]
fn test_clamped_plate_ir_matches_leissa_mode_ratios_within_two_point_five_percent() {
    let mut tool = ChladniWaveTankTool::default();
    tool.state.medium = Wave2DMedium::BiharmonicPlate;
    tool.state.boundary = BoundaryMode2D::Clamped;
    tool.state.clamp_layout = ClampLayout::None;
    tool.state.plate_stiffness = 0.0004;
    tool.state.damping = 0.0;
    tool.state.driver_x = 0.31;
    tool.state.driver_y = 0.23;
    tool.state.pickup_left = [0.69, 0.37];
    tool.state.pickup_right = [0.27, 0.73];
    tool.state.ir_sample_rate = WavSampleRate::Hz48000;
    tool.state.ir_format = WavSampleFormat::Float32;

    let wav = tool.render_complete_ir_blocking(0.75);
    let (sr, left, right) = decode_float32_stereo_wav(&wav);
    assert_eq!(sr, 48_000);
    assert_eq!(left.len(), 36_000);
    assert!(left.iter().all(|s| s.is_finite()));
    assert!(right.iter().all(|s| s.is_finite()));

    // Fundamental f_(1,1) sits near ~85-95 Hz on the 36x36 single-substep clamped plate grid.
    let f11 = find_parabolic_fft_peak_hz(&left, sr, 65.0, 125.0);
    let omega11 = analytic_clamped_square_omega(1, 1, 1.0, 0.0004);

    for (m, n) in [(1, 2), (2, 2), (1, 3)] {
        let expected_ratio = analytic_clamped_square_omega(m, n, 1.0, 0.0004) / omega11;
        let expected_hz = f11 * expected_ratio;
        let measured_hz =
            find_parabolic_fft_peak_hz(&left, sr, expected_hz * 0.88, expected_hz * 1.12);
        let measured_ratio = measured_hz / f11;
        let rel_err = ((measured_ratio - expected_ratio) / expected_ratio).abs();
        assert!(
            rel_err <= 0.025,
            "Clamped plate mode ({m},{n})/(1,1) ratio {measured_ratio:.4} vs Leissa {expected_ratio:.4} (err {:.2}%)",
            rel_err * 100.0
        );
    }
}

#[test]
fn test_ir_reflects_slit_barrier_and_refractive_lens_masks() {
    let mut tool = ChladniWaveTankTool::default();
    tool.apply_preset(TankPreset::DoubleSlitInterference);
    tool.state.driver_x = 0.14;
    tool.state.driver_y = 0.50;
    // Left pickup behind opaque barrier segment; right pickup aligned with slit aperture.
    tool.state.pickup_left = [0.32, 0.50];
    tool.state.pickup_right = [0.32, 0.66];
    tool.state.barrier_enabled = false;
    let (_, open_l, open_r) = decode_float32_stereo_wav(&tool.render_complete_ir_blocking(0.20));
    tool.state.barrier_enabled = true;
    let (_, slit_l, slit_r) = decode_float32_stereo_wav(&tool.render_complete_ir_blocking(0.20));

    let rms = |v: &[f32]| -> f64 {
        (v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / (v.len() as f64)).sqrt()
    };
    let open_balance = rms(&open_l) / rms(&open_r).max(1.0e-9);
    let slit_balance = rms(&slit_l) / rms(&slit_r).max(1.0e-9);
    assert!(
        (open_balance - slit_balance).abs() > 0.08,
        "Multi-slit barrier must alter L/R diffraction balance: open={open_balance:.3}, slit={slit_balance:.3}"
    );

    // Refractive biconvex lens delays wave transit time along the optical axis.
    tool.apply_preset(TankPreset::AcousticLens);
    tool.state.driver_x = 0.12;
    tool.state.driver_y = 0.50;
    tool.state.pickup_left = [0.68, 0.50];
    tool.state.refractive_index = 2.2;
    tool.state.lens_enabled = false;
    let (_, unlensed, _) = decode_float32_stereo_wav(&tool.render_complete_ir_blocking(0.20));
    tool.state.lens_enabled = true;
    let (_, lensed, _) = decode_float32_stereo_wav(&tool.render_complete_ir_blocking(0.20));

    let first_arrival = |sig: &[f32]| -> usize {
        let peak = sig.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
        sig.iter().position(|&v| v.abs() > 0.30 * peak).unwrap_or(0)
    };
    assert!(
        first_arrival(&lensed) > first_arrival(&unlensed) + 4,
        "Refractive lens (n=2.2) must delay wavefront arrival: unlensed={}, lensed={}",
        first_arrival(&unlensed),
        first_arrival(&lensed)
    );
}

#[test]
fn test_zero_alloc_pickup_stepping_and_cooperative_ir_export_with_url_roundtrip() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut tool = ChladniWaveTankTool {
        paused: true,
        ..Default::default()
    };
    tool.state.pickup_left = [0.337, 0.618]; // off-grid coordinates
    tool.state.pickup_right = [0.712, 0.289];
    tool.ir_studio
        .prepare_ir_capture(&tool.state, &sim::WorkbenchMaskContext::default(), 256);

    let (stepped, _metrics) = VerifiedEngine::run_verified(|| {
        tool.ir_studio.step_pickup_frames_zero_alloc(
            tool.state.pickup_left,
            tool.state.pickup_right,
            256,
        )
    });
    assert_eq!(stepped, 256);
    assert!(tool.ir_studio.left_buf[..256]
        .iter()
        .any(|&v| v.abs() > 1.0e-5));
    drop(_guard);

    // Cooperative UI slice rendering + strike audition + URL roundtrip.
    tool.state.ir_duration_secs = 0.5;
    tool.state.ir_sample_rate = WavSampleRate::Hz44100;
    tool.state.ir_format = WavSampleFormat::Pcm16;
    tool.state_rev = tool.state_rev.wrapping_add(1);
    tool.start_cooperative_ir_export(true);
    let ctx = Context::default();
    let raw = RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 800.0))),
        ..Default::default()
    };
    for _ in 0..450 {
        let _ = ctx.run(raw.clone(), |ctx| tool.show(ctx));
        if !tool.ir_studio.is_rendering {
            break;
        }
    }
    assert!(
        !tool.ir_studio.is_rendering,
        "cooperative IR render must finish"
    );
    let wav = tool
        .ir_studio
        .last_rendered_wav
        .as_ref()
        .expect("WAV cached");
    let summary = inspect_wav_riff(wav).expect("valid PCM16 44.1k WAV");
    assert_eq!(summary.format_tag, 1);
    assert_eq!(summary.sample_rate_hz, 44_100);

    tool.audition_strike_from_user_gesture();
    assert_eq!(snapshot_studio_preview().last_strike_sample_rate, 44_100);

    let json = tool.save_state().unwrap();
    let mut restored = ChladniWaveTankTool::default();
    restored.try_load_state(&json).unwrap();
    assert_eq!(restored.state.pickup_left, [0.337, 0.618]);
    assert_eq!(restored.state.pickup_right, [0.712, 0.289]);
    assert_eq!(restored.state.ir_sample_rate, WavSampleRate::Hz44100);
    assert_eq!(restored.state.ir_format, WavSampleFormat::Pcm16);
}

fn run_pickup_canvas_frame(
    ctx: &Context,
    events: Vec<eframe::egui::Event>,
    tool: &mut ChladniWaveTankTool,
) -> Rect {
    use eframe::egui::{CentralPanel, Sense, Vec2};
    let raw = RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(400.0, 400.0))),
        events,
        ..Default::default()
    };
    let mut canvas_rect = Rect::NOTHING;
    let _ = ctx.run(raw, |ctx| {
        CentralPanel::default().show(ctx, |ui| {
            let (resp, _painter) = ui.allocate_painter(Vec2::splat(200.0), Sense::click_and_drag());
            canvas_rect = resp.rect;
            tool.try_handle_pickup_drag(&resp, resp.rect);
        });
    });
    canvas_rect
}

fn drag_pickup_between(ctx: &Context, tool: &mut ChladniWaveTankTool, start: Pos2, end: Pos2) {
    use eframe::egui::{Event, Modifiers, PointerButton};
    run_pickup_canvas_frame(
        ctx,
        vec![
            Event::PointerMoved(start),
            Event::PointerButton {
                pos: start,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: Modifiers::NONE,
            },
        ],
        tool,
    );
    run_pickup_canvas_frame(ctx, vec![Event::PointerMoved(end)], tool);
    run_pickup_canvas_frame(
        ctx,
        vec![Event::PointerButton {
            pos: end,
            button: PointerButton::Primary,
            pressed: false,
            modifiers: Modifiers::NONE,
        }],
        tool,
    );
}

#[test]
fn test_draggable_stereo_pickups_left_and_right_on_canvas() {
    let mut tool = ChladniWaveTankTool {
        paused: true,
        ..Default::default()
    };
    tool.state.pickup_left = [0.25, 0.75];
    tool.state.pickup_right = [0.75, 0.25];
    let ctx = Context::default();
    let canvas_rect = run_pickup_canvas_frame(&ctx, Vec::new(), &mut tool);
    let to_screen = |nx: f64, ny: f64| -> Pos2 {
        Pos2::new(
            canvas_rect.left() + (nx as f32) * canvas_rect.width(),
            canvas_rect.bottom() - (ny as f32) * canvas_rect.height(),
        )
    };

    drag_pickup_between(
        &ctx,
        &mut tool,
        to_screen(0.25, 0.75),
        to_screen(0.40, 0.60),
    );
    assert!((tool.state.pickup_left[0] - 0.40).abs() < 0.02);
    assert!((tool.state.pickup_left[1] - 0.60).abs() < 0.02);
    assert!((tool.state.pickup_right[0] - 0.75).abs() < 1.0e-6);

    drag_pickup_between(
        &ctx,
        &mut tool,
        to_screen(0.75, 0.25),
        to_screen(0.82, 0.48),
    );
    assert!((tool.state.pickup_right[0] - 0.82).abs() < 0.02);
    assert!((tool.state.pickup_right[1] - 0.48).abs() < 0.02);
}

#[test]
fn test_sculpted_clamp_and_spruce_rib_masks_propagate_into_stereo_ir() {
    let mut tool = ChladniWaveTankTool::default();
    tool.apply_preset(TankPreset::SquareChladniPlate);
    tool.state.driver_x = 0.25;
    tool.state.driver_y = 0.50;
    tool.state.pickup_left = [0.55, 0.50];
    tool.state.pickup_right = [0.55, 0.78];

    let (_, base_l, _) = decode_float32_stereo_wav(&tool.render_complete_ir_blocking(0.20));

    for gy in 48..80 {
        for gx in 56..72 {
            let idx = math_explorer::math_kernel::types::flatten_2d_index(gx, gy, GRID_SIZE);
            tool.sculpt_mask[idx] = sculpt::AcousticCellRole::RigidClamp as u8;
        }
        for gx in 84..96 {
            let idx = math_explorer::math_kernel::types::flatten_2d_index(gx, gy, GRID_SIZE);
            tool.sculpt_mask[idx] = sculpt::AcousticCellRole::SpruceRib as u8;
        }
    }
    tool.state_rev = tool.state_rev.wrapping_add(1);
    let (_, sculpted_l, sculpted_r) =
        decode_float32_stereo_wav(&tool.render_complete_ir_blocking(0.20));
    assert!(sculpted_l.iter().all(|s| s.is_finite()));
    assert!(sculpted_r.iter().all(|s| s.is_finite()));
    let max_diff = base_l
        .iter()
        .zip(sculpted_l.iter())
        .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        max_diff > 0.05,
        "Sculpted rigid clamps and spruce ribs must alter rendered plate IR: max_diff={max_diff}"
    );
}
