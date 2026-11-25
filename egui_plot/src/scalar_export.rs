//! 16-bit grayscale heightmap PNG and central-difference PBR tangent-space normal-map PNG encoders.

use crate::export::MAX_EXPORT_DIMENSION;
use math_commons::math_kernel::types::flatten_2d_index;

fn validate_scalar_grid(values: &[f64], width: usize, height: usize) -> Result<usize, String> {
    if width == 0 || height == 0 {
        return Err("Scalar field dimensions must be non-zero".to_string());
    }
    let max_dim = MAX_EXPORT_DIMENSION as usize;
    if width > max_dim || height > max_dim {
        return Err(format!(
            "Scalar field dimensions ({width}x{height}) exceed maximum limit of {max_dim}x{max_dim}"
        ));
    }
    let len = width
        .checked_mul(height)
        .ok_or_else(|| "Scalar field dimensions overflowed buffer size".to_string())?;
    if values.len() != len {
        return Err(format!(
            "Scalar field length ({}) does not match width * height ({len})",
            values.len()
        ));
    }
    Ok(len)
}

fn finite_min_max(values: &[f64]) -> (f64, f64) {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    if !min_v.is_finite() || !max_v.is_finite() {
        (0.0, 0.0)
    } else {
        (min_v, max_v)
    }
}

fn normalize_scalar_slice(values: &[f64]) -> Vec<f64> {
    let (min_v, max_v) = finite_min_max(values);
    let span = max_v - min_v;
    if span.abs() <= 1.0e-15 {
        let uniform = min_v.clamp(0.0, 1.0);
        return vec![uniform; values.len()];
    }
    let inv_span = 1.0 / span;
    values
        .iter()
        .map(|&v| {
            if v.is_finite() {
                ((v - min_v) * inv_span).clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect()
}

/// Encodes a 2D `f64` scalar grid into big-endian 16-bit grayscale PNG bytes
/// (`png::ColorType::Grayscale`, `png::BitDepth::Sixteen`, `0..=65535`).
pub fn encode_heightmap_png_16bit(
    values: &[f64],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, String> {
    let len = validate_scalar_grid(values, width, height)?;
    let normalized = normalize_scalar_slice(values);
    let mut raw_be = Vec::with_capacity(len * 2);
    for &t in &normalized {
        let sample = (t * 65535.0).round().clamp(0.0, 65535.0) as u16;
        raw_be.extend_from_slice(&sample.to_be_bytes());
    }

    let mut png_bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut png_bytes, width as u32, height as u32);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Sixteen);
    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
    writer
        .write_image_data(&raw_be)
        .map_err(|e| e.to_string())?;
    drop(writer);
    Ok(png_bytes)
}

/// Decodes a 16-bit grayscale PNG into `(width, height, u16_samples)` while verifying
/// `png::ColorType::Grayscale` and `png::BitDepth::Sixteen` header metadata.
pub fn decode_heightmap_png_16bit(png_bytes: &[u8]) -> Result<(usize, usize, Vec<u16>), String> {
    let decoder = png::Decoder::new(std::io::Cursor::new(png_bytes));
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("Failed to read PNG header: {e}"))?;
    let info = reader.info();
    if info.color_type != png::ColorType::Grayscale || info.bit_depth != png::BitDepth::Sixteen {
        return Err(format!(
            "Expected 16-bit Grayscale PNG, got {:?} / {:?}",
            info.color_type, info.bit_depth
        ));
    }
    let (width, height) = (info.width as usize, info.height as usize);
    let buf_size = reader
        .output_buffer_size()
        .ok_or_else(|| "PNG output buffer size overflow".to_string())?;
    let mut buf = vec![0_u8; buf_size];
    let frame = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("Failed to decode 16-bit PNG frame: {e}"))?;
    let slice = &buf[..frame.buffer_size()];
    let samples: Vec<u16> = slice
        .as_chunks::<2>()
        .0
        .iter()
        .copied()
        .map(u16::from_be_bytes)
        .collect();
    Ok((width, height, samples))
}

#[inline]
fn unit_normal_component_to_u8(val: f64) -> u8 {
    ((val * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8
}

#[inline]
fn axis_neighbors(coord: usize, extent: usize, periodic: bool) -> (usize, usize, f64) {
    if extent <= 1 {
        return (0, 0, 1.0);
    }
    if periodic {
        let prev = if coord > 0 { coord - 1 } else { extent - 1 };
        let next = if coord + 1 < extent { coord + 1 } else { 0 };
        (prev, next, 2.0)
    } else {
        let prev = coord.saturating_sub(1);
        let next = (coord + 1).min(extent - 1);
        (prev, next, ((next - prev) as f64).max(1.0))
    }
}

/// Encodes a 2D `f64` scalar heightfield into an 8-bit RGB tangent-space PBR normal-map PNG
/// using central differences (wrapping modulo `(width, height)` when `periodic` is `true`).
pub fn encode_normal_map_png(
    values: &[f64],
    width: usize,
    height: usize,
    strength: f64,
    periodic: bool,
) -> Result<Vec<u8>, String> {
    let len = validate_scalar_grid(values, width, height)?;
    let normalized = normalize_scalar_slice(values);
    let amp = if strength.is_finite() {
        strength.clamp(0.0, 256.0)
    } else {
        4.0
    };
    let mut rgb_bytes = Vec::with_capacity(len * 3);

    for y in 0..height {
        let (y_u, y_d, dy) = axis_neighbors(y, height, periodic);
        for x in 0..width {
            let (x_l, x_r, dx) = axis_neighbors(x, width, periodic);
            let h_l = normalized[flatten_2d_index(x_l, y, width)];
            let h_r = normalized[flatten_2d_index(x_r, y, width)];
            let h_u = normalized[flatten_2d_index(x, y_u, width)];
            let h_d = normalized[flatten_2d_index(x, y_d, width)];

            let dz_dx = (h_r - h_l) / dx;
            let dz_dy = (h_d - h_u) / dy;
            let nx_raw = -dz_dx * amp;
            let ny_raw = dz_dy * amp;
            let nz_raw = 1.0;
            let inv_norm = 1.0 / (nx_raw * nx_raw + ny_raw * ny_raw + nz_raw * nz_raw).sqrt();

            rgb_bytes.push(unit_normal_component_to_u8(nx_raw * inv_norm));
            rgb_bytes.push(unit_normal_component_to_u8(ny_raw * inv_norm));
            rgb_bytes.push(unit_normal_component_to_u8(nz_raw * inv_norm));
        }
    }

    let mut png_bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut png_bytes, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
    writer
        .write_image_data(&rgb_bytes)
        .map_err(|e| e.to_string())?;
    drop(writer);
    Ok(png_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export_primitives::decode_png_rgba;
    use std::f64::consts::TAU;

    #[test]
    fn test_heightmap_png_16bit_header_and_monotonic_dynamic_range() {
        let values: Vec<f64> = (0..64).map(|i| -5.0 + (i as f64) * 0.25).collect();
        let png = encode_heightmap_png_16bit(&values, 8, 8).expect("16-bit heightmap encode");
        let (w, h, decoded) = decode_heightmap_png_16bit(&png).expect("16-bit heightmap decode");
        assert_eq!((w, h), (8, 8));
        assert_eq!(decoded[0], 0);
        assert_eq!(decoded[63], 65535);
        for pair in decoded.windows(2) {
            assert!(
                pair[1] > pair[0],
                "Expected strictly monotonic 16-bit values"
            );
        }
    }

    #[test]
    fn test_normal_map_periodic_wraps_seamlessly_across_opposite_borders() {
        let (w, h) = (32, 32);
        let mut field = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let fx = TAU * (x as f64) / (w as f64);
                let fy = TAU * (y as f64) / (h as f64);
                field[flatten_2d_index(x, y, w)] = fx.sin() * fy.cos();
            }
        }
        // Shift field by (w/2, h/2); periodic normal map shifted back must match at seams!
        let mut shifted = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                shifted[flatten_2d_index(x, y, w)] =
                    field[flatten_2d_index((x + w / 2) % w, (y + h / 2) % h, w)];
            }
        }
        let png_orig = encode_normal_map_png(&field, w, h, 4.0, true).expect("periodic normal map");
        let png_shift =
            encode_normal_map_png(&shifted, w, h, 4.0, true).expect("shifted normal map");
        let (_, _, px_orig) = decode_png_rgba(&png_orig).expect("decode orig");
        let (_, _, px_shift) = decode_png_rgba(&png_shift).expect("decode shift");

        for y in 0..h {
            for x in 0..w {
                let a = px_orig[flatten_2d_index((x + w / 2) % w, (y + h / 2) % h, w)];
                let b = px_shift[flatten_2d_index(x, y, w)];
                assert_eq!(
                    a, b,
                    "Toroidal normal map mismatch across seam at ({x}, {y})"
                );
                assert!(b[2] >= 128, "Expected positive hemisphere Z normal");
            }
        }
    }
}
