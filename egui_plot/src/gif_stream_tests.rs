use super::*;
use crate::export::{ExportOptions, export_custom_gif, sample_progressive_frames};
use crate::export_primitives::export_raster_gif;
use image::AnimationDecoder;

fn has_netscape_infinite_loop_extension(gif_bytes: &[u8]) -> bool {
    const MARKER: &[u8] = b"NETSCAPE2.0\x03\x01\x00\x00\x00";
    gif_bytes.windows(MARKER.len()).any(|w| w == MARKER)
}

#[test]
fn test_gif_speed_clamping_and_default_ten() {
    assert_eq!(DEFAULT_GIF_SPEED, 10);
    assert_eq!(clamp_gif_speed(-10), MIN_GIF_SPEED);
    assert_eq!(clamp_gif_speed(0), MIN_GIF_SPEED);
    assert_eq!(clamp_gif_speed(10), 10);
    assert_eq!(clamp_gif_speed(30), MAX_GIF_SPEED);
    assert_eq!(clamp_gif_speed(99), MAX_GIF_SPEED);

    let default_enc = StreamingGifEncoder::new(15).expect("default encoder");
    assert_eq!(default_enc.speed(), 10);
    assert_eq!(default_enc.frames_encoded(), 0);

    let clamped_high = StreamingGifEncoder::new_with_speed(15, 100).expect("clamped encoder");
    assert_eq!(clamped_high.speed(), 30);

    let clamped_low = StreamingGifEncoder::new_with_speed(15, -3).expect("clamped low encoder");
    assert_eq!(clamped_low.speed(), 1);
}

#[test]
fn test_streaming_gif_decodes_valid_animation_infinite_repeat_dimensions_and_fps() {
    let datasets = vec![(
        "Oscillator".to_string(),
        vec![[0.0, 0.0], [1.0, 1.0], [2.0, -0.5], [3.0, 0.8], [4.0, 0.2]],
    )];
    let sampled = sample_progressive_frames(&datasets, 5);
    let opts = ExportOptions {
        width: 320,
        height: 180,
        ..Default::default()
    };
    let gif_bytes = export_custom_gif(&sampled, &opts, 10).expect("custom GIF");
    assert!(gif_bytes.starts_with(b"GIF89a"));
    assert!(
        has_netscape_infinite_loop_extension(&gif_bytes),
        "GIF must contain NETSCAPE2.0 Repeat::Infinite block"
    );

    let decoder =
        image::codecs::gif::GifDecoder::new(std::io::Cursor::new(&gif_bytes)).expect("decode GIF");
    let decoded_frames: Vec<image::Frame> = decoder
        .into_frames()
        .collect_frames()
        .expect("collect GIF frames");
    assert_eq!(decoded_frames.len(), 5);
    for frame in &decoded_frames {
        assert_eq!(frame.buffer().dimensions(), (320, 180));
        let (numer, denom) = frame.delay().numer_denom_ms();
        let delay_ms = numer / denom.max(1);
        assert_eq!(delay_ms, 100, "10 FPS must encode 100ms frame delay");
    }
}

#[test]
fn test_raster_gif_streams_scaled_frames_with_target_dimensions_and_fps_delay() {
    let mut raw_frames = Vec::new();
    for step in 0..4_u8 {
        let pixels = vec![egui::Color32::from_rgb(20 + step * 50, 80, 200 - step * 30); 16 * 12];
        raw_frames.push(egui::ColorImage::new([16, 12], pixels));
    }
    let opts = ExportOptions {
        width: 240,
        height: 240,
        aspect_ratio: crate::export::AspectRatioOption::OneOne,
        ..Default::default()
    };
    let gif_bytes = export_raster_gif(&raw_frames, &opts, 25).expect("raster GIF");
    assert!(gif_bytes.starts_with(b"GIF89a"));
    assert!(has_netscape_infinite_loop_extension(&gif_bytes));

    let decoder =
        image::codecs::gif::GifDecoder::new(std::io::Cursor::new(&gif_bytes)).expect("decode GIF");
    let decoded_frames = decoder
        .into_frames()
        .collect_frames()
        .expect("collect frames");
    assert_eq!(decoded_frames.len(), 4);
    for frame in &decoded_frames {
        assert_eq!(frame.buffer().dimensions(), (240, 240));
        let (numer, denom) = frame.delay().numer_denom_ms();
        assert_eq!(numer / denom.max(1), 40, "25 FPS must encode 40ms delay");
    }
}
