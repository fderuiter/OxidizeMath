//! Streaming single-frame animated GIF encoding utilities.

use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Default NeuQuant palette quantization speed (`1..=30`) for [`StreamingGifEncoder`].
pub const DEFAULT_GIF_SPEED: i32 = 10;

/// Minimum allowed GIF palette quantization speed (slowest, highest palette search effort).
pub const MIN_GIF_SPEED: i32 = 1;

/// Maximum allowed GIF palette quantization speed (fastest quantization).
pub const MAX_GIF_SPEED: i32 = 30;

/// Clamp a requested GIF encoder `speed` parameter into the valid `1..=30` range.
#[must_use]
pub fn clamp_gif_speed(speed: i32) -> i32 {
    speed.clamp(MIN_GIF_SPEED, MAX_GIF_SPEED)
}

/// Compute the per-frame [`image::Delay`] corresponding to `fps` clamped to `1..=60`.
#[must_use]
pub fn gif_frame_delay(fps: u32) -> image::Delay {
    let clamped_fps = fps.clamp(1, 60);
    let delay_ms = (1000 / clamped_fps).max(1);
    image::Delay::from_saturating_duration(Duration::from_millis(u64::from(delay_ms)))
}

#[derive(Clone)]
struct SharedByteWriter(Arc<Mutex<Vec<u8>>>);

impl Write for SharedByteWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut guard = self.0.lock().unwrap_or_else(|p| p.into_inner());
        guard.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Incremental animated GIF encoder that streams one [`egui::ColorImage`] frame at a time.
///
/// Each frame is packed into an [`image::RgbaImage`] via [`egui::Color32::to_array`], the
/// input [`egui::ColorImage`] is dropped immediately, and the [`image::Frame`] is encoded
/// via [`image::codecs::gif::GifEncoder::new_with_speed`] with [`image::codecs::gif::Repeat::Infinite`].
pub struct StreamingGifEncoder {
    encoder: Option<image::codecs::gif::GifEncoder<SharedByteWriter>>,
    buffer: Arc<Mutex<Vec<u8>>>,
    delay: image::Delay,
    speed: i32,
    frames_encoded: usize,
}

impl std::fmt::Debug for StreamingGifEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingGifEncoder")
            .field("speed", &self.speed)
            .field("frames_encoded", &self.frames_encoded)
            .finish()
    }
}

impl StreamingGifEncoder {
    /// Construct a streaming GIF encoder at `fps` using [`DEFAULT_GIF_SPEED`] (`10`).
    pub fn new(fps: u32) -> Result<Self, String> {
        Self::new_with_speed(fps, DEFAULT_GIF_SPEED)
    }

    /// Construct a streaming GIF encoder at `fps` with `speed` clamped to `1..=30`.
    pub fn new_with_speed(fps: u32, speed: i32) -> Result<Self, String> {
        let clamped_speed = clamp_gif_speed(speed);
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let writer = SharedByteWriter(Arc::clone(&buffer));
        let mut encoder = image::codecs::gif::GifEncoder::new_with_speed(writer, clamped_speed);
        encoder
            .set_repeat(image::codecs::gif::Repeat::Infinite)
            .map_err(|e| format!("Failed to set GIF repeat mode: {e}"))?;
        Ok(Self {
            encoder: Some(encoder),
            buffer,
            delay: gif_frame_delay(fps),
            speed: clamped_speed,
            frames_encoded: 0,
        })
    }

    /// Return the clamped palette quantization speed (`1..=30`) configured on this encoder.
    #[must_use]
    pub const fn speed(&self) -> i32 {
        self.speed
    }

    /// Return the number of non-empty frames encoded so far.
    #[must_use]
    pub const fn frames_encoded(&self) -> usize {
        self.frames_encoded
    }

    /// Pack an owned [`egui::ColorImage`], drop it, and stream the frame into the GIF output.
    ///
    /// Returns `Ok(false)` when `image` has zero width or height and was skipped.
    pub fn encode_color_image(&mut self, image: egui::ColorImage) -> Result<bool, String> {
        let width = image.width() as u32;
        let height = image.height() as u32;
        if width == 0 || height == 0 {
            return Ok(false);
        }
        let rgba_buf = pack_color_image_rgba(&image);
        drop(image);
        self.encode_rgba_buffer(width, height, rgba_buf)?;
        Ok(true)
    }

    /// Pack a borrowed [`egui::ColorImage`] slice entry and stream the frame into the GIF output.
    pub fn encode_color_image_ref(&mut self, image: &egui::ColorImage) -> Result<bool, String> {
        let width = image.width() as u32;
        let height = image.height() as u32;
        if width == 0 || height == 0 {
            return Ok(false);
        }
        let rgba_buf = pack_color_image_rgba(image);
        self.encode_rgba_buffer(width, height, rgba_buf)?;
        Ok(true)
    }

    fn encode_rgba_buffer(
        &mut self,
        width: u32,
        height: u32,
        rgba_buf: Vec<u8>,
    ) -> Result<(), String> {
        let rgba_img = image::RgbaImage::from_raw(width, height, rgba_buf)
            .ok_or_else(|| "Failed to construct RGBA buffer for GIF frame".to_string())?;
        let frame = image::Frame::from_parts(rgba_img, 0, 0, self.delay);
        let enc = self
            .encoder
            .as_mut()
            .ok_or_else(|| "GIF encoder already finalized".to_string())?;
        enc.encode_frame(frame)
            .map_err(|e| format!("Failed to encode GIF frame: {e}"))?;
        self.frames_encoded += 1;
        Ok(())
    }

    /// Finalize the GIF trailer (`0x3B`) and extract the encoded GIF byte vector.
    pub fn finish(mut self) -> Result<Vec<u8>, String> {
        drop(self.encoder.take());
        if self.frames_encoded == 0 {
            return Err("GIF encoding produced empty byte stream".to_string());
        }
        let gif_bytes = match Arc::try_unwrap(self.buffer) {
            Ok(mutex) => mutex.into_inner().unwrap_or_else(|p| p.into_inner()),
            Err(arc) => std::mem::take(&mut *arc.lock().unwrap_or_else(|p| p.into_inner())),
        };
        if gif_bytes.is_empty() {
            return Err("GIF encoding produced empty byte stream".to_string());
        }
        Ok(gif_bytes)
    }
}

pub(crate) fn pack_color_image_rgba(image: &egui::ColorImage) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(image.pixels.len() * 4);
    for pixel in &image.pixels {
        rgba.extend_from_slice(&pixel.to_array());
    }
    rgba
}

/// Encode borrowed [`egui::ColorImage`] frames into an animated GIF with configurable `speed` (`1..=30`).
pub fn color32_images_to_gif_with_speed(
    images: &[egui::ColorImage],
    fps: u32,
    speed: i32,
) -> Result<Vec<u8>, String> {
    if images.is_empty() {
        return Err("No frames provided for GIF encoding".to_string());
    }
    let mut encoder = StreamingGifEncoder::new_with_speed(fps, speed)?;
    for img in images {
        encoder.encode_color_image_ref(img)?;
    }
    encoder.finish()
}

/// Decoded animation metadata and first-frame RGBA pixels from an encoded GIF byte stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GifAnimationSummary {
    /// Canvas pixel width of the decoded animation.
    pub width: u32,
    /// Canvas pixel height of the decoded animation.
    pub height: u32,
    /// Total number of animation frames decoded.
    pub frame_count: usize,
    /// Frame delay in milliseconds recorded on the first frame.
    pub first_delay_ms: u32,
    /// Whether the `NETSCAPE2.0` infinite-repeat application extension is present.
    pub infinite_repeat: bool,
    /// RGBA pixels of the first decoded frame (`width * height` entries).
    pub first_frame_rgba: Vec<[u8; 4]>,
}

/// Decode an animated GIF byte stream to verify dimensions, frame count, loop mode, and frame delay.
pub fn decode_gif_animation_summary(gif_bytes: &[u8]) -> Result<GifAnimationSummary, String> {
    use image::AnimationDecoder as _;

    if !gif_bytes.starts_with(b"GIF89a") {
        return Err("Missing GIF89a header".to_string());
    }
    const LOOP_MARKER: &[u8] = b"NETSCAPE2.0\x03\x01\x00\x00\x00";
    let infinite_repeat = gif_bytes
        .windows(LOOP_MARKER.len())
        .any(|w| w == LOOP_MARKER);
    let decoder = image::codecs::gif::GifDecoder::new(std::io::Cursor::new(gif_bytes))
        .map_err(|e| format!("Failed to parse GIF stream: {e}"))?;
    let frames = decoder
        .into_frames()
        .collect_frames()
        .map_err(|e| format!("Failed to decode GIF frames: {e}"))?;
    let first = frames
        .first()
        .ok_or_else(|| "Decoded GIF contained zero frames".to_string())?;
    let (width, height) = first.buffer().dimensions();
    let (numer, denom) = first.delay().numer_denom_ms();
    let first_delay_ms = numer / denom.max(1);
    let first_frame_rgba = first.buffer().pixels().map(|p| p.0).collect();
    Ok(GifAnimationSummary {
        width,
        height,
        frame_count: frames.len(),
        first_delay_ms,
        infinite_repeat,
        first_frame_rgba,
    })
}

#[cfg(test)]
#[path = "gif_stream_tests.rs"]
mod tests;
