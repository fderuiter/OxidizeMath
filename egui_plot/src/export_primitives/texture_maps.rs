//! Borderless power-of-two 16-bit grayscale heightmap and tangent-space normal-map PNG encoders.

/// Supported borderless power-of-two square resolutions for 3D material exports.
pub const TEXTURE_SIZE_PRESETS: [u32; 4] = [256, 512, 1024, 2048];

/// Validates that `size` is one of [`TEXTURE_SIZE_PRESETS`] (`256`, `512`, `1024`, `2048`).
pub fn sanitize_texture_size(size: u32) -> Result<usize, String> {
    if TEXTURE_SIZE_PRESETS.contains(&size) {
        Ok(size as usize)
    } else {
        Err(format!(
            "Unsupported texture resolution {size}x{size}; expected 256, 512, 1024, or 2048"
        ))
    }
}

fn finite_range(values: &[f64]) -> Result<(f64, f64), String> {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    let mut any_finite = false;
    for &v in values {
        if v.is_finite() {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
            any_finite = true;
        }
    }
    if any_finite {
        Ok((min_v, max_v))
    } else {
        Err("Scalar field contains no finite samples for texture export".to_string())
    }
}

#[inline]
fn sample_sanitized(values: &[f64], x: usize, y: usize, width: usize, fallback: f64) -> f64 {
    let val = values[y * width + x];
    if val.is_finite() { val } else { fallback }
}

/// Resamples a 2D scalar field to an exact borderless `dst_w x dst_h` buffer using bilinear interpolation.
///
/// When `periodic` is `true`, coordinates wrap across `[0, src_w)` and `[0, src_h)` toroidally
/// so opposite edges tile without seams. When `false`, edge samples are clamped without border padding.
pub fn resample_scalar_field_borderless(
    values: &[f64],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    periodic: bool,
) -> Result<Vec<f64>, String> {
    if src_w == 0 || src_h == 0 || values.len() < src_w * src_h {
        return Err(
            "Source scalar field dimensions are invalid or buffer is truncated".to_string(),
        );
    }
    if dst_w == 0 || dst_h == 0 {
        return Err("Target texture dimensions must be non-zero".to_string());
    }
    let (min_v, _) = finite_range(&values[..src_w * src_h])?;
    if src_w == dst_w && src_h == dst_h {
        return Ok(values[..src_w * src_h]
            .iter()
            .map(|&v| if v.is_finite() { v } else { min_v })
            .collect());
    }

    let mut out = vec![0.0_f64; dst_w * dst_h];
    for dy in 0..dst_h {
        let (y0, y1, ty) = axis_interp(dy, dst_h, src_h, periodic);
        for dx in 0..dst_w {
            let (x0, x1, tx) = axis_interp(dx, dst_w, src_w, periodic);
            let v00 = sample_sanitized(values, x0, y0, src_w, min_v);
            let v10 = sample_sanitized(values, x1, y0, src_w, min_v);
            let v01 = sample_sanitized(values, x0, y1, src_w, min_v);
            let v11 = sample_sanitized(values, x1, y1, src_w, min_v);
            let top = v00 + tx * (v10 - v00);
            let bot = v01 + tx * (v11 - v01);
            out[dy * dst_w + dx] = top + ty * (bot - top);
        }
    }
    Ok(out)
}

#[inline]
fn axis_interp(
    dst_idx: usize,
    dst_len: usize,
    src_len: usize,
    periodic: bool,
) -> (usize, usize, f64) {
    if src_len <= 1 {
        return (0, 0, 0.0);
    }
    if periodic {
        let coord = (dst_idx as f64 / dst_len as f64) * (src_len as f64);
        let base = coord.floor();
        let i0 = (base as usize) % src_len;
        let i1 = (i0 + 1) % src_len;
        (i0, i1, (coord - base).clamp(0.0, 1.0))
    } else {
        let denom = dst_len.saturating_sub(1).max(1) as f64;
        let coord = (dst_idx as f64 / denom) * ((src_len - 1) as f64);
        let base = coord.floor();
        let i0 = (base as usize).min(src_len - 1);
        let i1 = (i0 + 1).min(src_len - 1);
        (i0, i1, (coord - base).clamp(0.0, 1.0))
    }
}

/// Encodes a 2D scalar field as a borderless power-of-two 16-bit grayscale PNG (`ColorType::Grayscale`, `BitDepth::Sixteen`).
pub fn encode_heightmap_16bit_png(
    values: &[f64],
    src_w: usize,
    src_h: usize,
    size: u32,
    periodic: bool,
) -> Result<Vec<u8>, String> {
    let dst = sanitize_texture_size(size)?;
    let resampled = resample_scalar_field_borderless(values, src_w, src_h, dst, dst, periodic)?;
    let (min_v, max_v) = finite_range(&resampled)?;
    let span = max_v - min_v;

    let mut raw_be = Vec::with_capacity(dst * dst * 2);
    for &v in &resampled {
        let norm = if span > 1e-15 {
            ((v - min_v) / span).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let q = (norm * 65535.0).round() as u16;
        raw_be.extend_from_slice(&q.to_be_bytes());
    }

    let mut png_bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut png_bytes, size, size);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Sixteen);
    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
    writer
        .write_image_data(&raw_be)
        .map_err(|e| e.to_string())?;
    drop(writer);
    Ok(png_bytes)
}

/// Encodes a tangent-space normal-map PNG (`ColorType::Rgb`, `BitDepth::Sixteen`) using periodic or clamped central differences.
pub fn encode_normal_map_png(
    values: &[f64],
    src_w: usize,
    src_h: usize,
    size: u32,
    periodic: bool,
    strength: f32,
) -> Result<Vec<u8>, String> {
    let dst = sanitize_texture_size(size)?;
    let resampled = resample_scalar_field_borderless(values, src_w, src_h, dst, dst, periodic)?;
    let (min_v, max_v) = finite_range(&resampled)?;
    let span = max_v - min_v;
    let heights: Vec<f64> = resampled
        .iter()
        .map(|&v| {
            if span > 1e-15 {
                ((v - min_v) / span).clamp(0.0, 1.0)
            } else {
                0.5
            }
        })
        .collect();

    let grad_scale = f64::from(strength.clamp(0.05, 64.0)) * (dst as f64 / 64.0);
    let mut raw_be = Vec::with_capacity(dst * dst * 6);
    for y in 0..dst {
        let y_prev = if y > 0 {
            y - 1
        } else if periodic {
            dst - 1
        } else {
            0
        };
        let y_next = if y + 1 < dst {
            y + 1
        } else if periodic {
            0
        } else {
            dst - 1
        };
        for x in 0..dst {
            let x_prev = if x > 0 {
                x - 1
            } else if periodic {
                dst - 1
            } else {
                0
            };
            let x_next = if x + 1 < dst {
                x + 1
            } else if periodic {
                0
            } else {
                dst - 1
            };
            let dh_dx = 0.5 * (heights[y * dst + x_next] - heights[y * dst + x_prev]) * grad_scale;
            let dh_dy = 0.5 * (heights[y_next * dst + x] - heights[y_prev * dst + x]) * grad_scale;
            let nx = -dh_dx;
            let ny = -dh_dy;
            let nz = 1.0_f64;
            let inv_len = 1.0 / (nx * nx + ny * ny + nz * nz).sqrt();
            for comp in [nx * inv_len, ny * inv_len, nz * inv_len] {
                let q = ((comp * 0.5 + 0.5).clamp(0.0, 1.0) * 65535.0).round() as u16;
                raw_be.extend_from_slice(&q.to_be_bytes());
            }
        }
    }

    let mut png_bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut png_bytes, size, size);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Sixteen);
    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
    writer
        .write_image_data(&raw_be)
        .map_err(|e| e.to_string())?;
    drop(writer);
    Ok(png_bytes)
}

/// Decodes a 16-bit grayscale heightmap PNG with `png::Decoder`, returning `(width, height, samples_u16)`.
pub fn decode_heightmap_16bit_png(png_bytes: &[u8]) -> Result<(u32, u32, Vec<u16>), String> {
    let decoder = png::Decoder::new(std::io::Cursor::new(png_bytes));
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("PNG decode error: {e}"))?;
    let info = reader.info();
    if info.color_type != png::ColorType::Grayscale || info.bit_depth != png::BitDepth::Sixteen {
        return Err(format!(
            "Expected 16-bit Grayscale PNG, got {:?} {:?}",
            info.color_type, info.bit_depth
        ));
    }
    let (width, height) = (info.width, info.height);
    let mut buf = vec![
        0_u8;
        reader
            .output_buffer_size()
            .ok_or("Missing PNG buffer size")?
    ];
    let frame = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("PNG frame error: {e}"))?;
    let bytes = &buf[..frame.buffer_size()];
    let samples = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .copied()
        .map(u16::from_be_bytes)
        .collect();
    Ok((width, height, samples))
}

/// Decodes a 16-bit RGB normal-map PNG with `png::Decoder`, returning `(width, height, rgb_u16_triplets)`.
pub fn decode_normal_map_16bit_png(png_bytes: &[u8]) -> Result<(u32, u32, Vec<[u16; 3]>), String> {
    let decoder = png::Decoder::new(std::io::Cursor::new(png_bytes));
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("PNG decode error: {e}"))?;
    let info = reader.info();
    if info.color_type != png::ColorType::Rgb || info.bit_depth != png::BitDepth::Sixteen {
        return Err(format!(
            "Expected 16-bit RGB normal map PNG, got {:?} {:?}",
            info.color_type, info.bit_depth
        ));
    }
    let (width, height) = (info.width, info.height);
    let mut buf = vec![
        0_u8;
        reader
            .output_buffer_size()
            .ok_or("Missing PNG buffer size")?
    ];
    let frame = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("PNG frame error: {e}"))?;
    let bytes = &buf[..frame.buffer_size()];
    let triplets = bytes
        .as_chunks::<6>()
        .0
        .iter()
        .map(|&[r0, r1, g0, g1, b0, b1]| {
            [
                u16::from_be_bytes([r0, r1]),
                u16::from_be_bytes([g0, g1]),
                u16::from_be_bytes([b0, b1]),
            ]
        })
        .collect();
    Ok((width, height, triplets))
}
