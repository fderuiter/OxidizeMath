//! Physical millimeter SVG serializer for laser cutters (`#layer-cut`) and pen plotters (`#layer-score`).
//!
//! Unlike [`super::svg::export_primitives_svg`], which targets screen/publication figures with
//! unitless pixel viewports and an opaque background `<rect>`, [`export_fabrication_svg`] writes
//! exact millimeter dimensions (`width="{w}mm" height="{h}mm" viewBox="0 0 {w} {h}"`), omits all
//! background `<rect>` and `<image>` elements, and groups pure `fill="none"` vector `<path>`
//! elements into `<g id="layer-score">` and `<g id="layer-cut">`.

/// Minimum custom physical fabrication bed dimension in millimeters.
pub const MIN_FABRICATION_BED_MM: f64 = 20.0;

/// Maximum custom physical fabrication bed dimension in millimeters.
pub const MAX_FABRICATION_BED_MM: f64 = 1200.0;

/// Standard or custom physical stock / bed dimensions in millimeters.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FabricationBedPreset {
    /// ISO A4 sheet (`210` × `297` mm).
    #[default]
    A4,
    /// ANSI US Letter sheet (`215.9` × `279.4` mm).
    UsLetter,
    /// Custom laser bed or plotter sheet (`20` to `1200` mm on each axis).
    Custom {
        /// Physical bed width in millimeters (`20.0..=1200.0`).
        width_mm: f64,
        /// Physical bed height in millimeters (`20.0..=1200.0`).
        height_mm: f64,
    },
}

impl FabricationBedPreset {
    /// Short display label for UI combo boxes.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::A4 => "A4 (210 × 297 mm)",
            Self::UsLetter => "US Letter (215.9 × 279.4 mm)",
            Self::Custom { .. } => "Custom Bed (mm)",
        }
    }

    /// Validates and returns exact `(width_mm, height_mm)` physical bed dimensions.
    pub fn dimensions_mm(&self) -> Result<(f64, f64), String> {
        match *self {
            Self::A4 => Ok((210.0, 297.0)),
            Self::UsLetter => Ok((215.9, 279.4)),
            Self::Custom {
                width_mm,
                height_mm,
            } => {
                if !width_mm.is_finite() || !height_mm.is_finite() {
                    return Err("Custom bed dimensions must be finite millimeters".to_string());
                }
                if !(MIN_FABRICATION_BED_MM..=MAX_FABRICATION_BED_MM).contains(&width_mm)
                    || !(MIN_FABRICATION_BED_MM..=MAX_FABRICATION_BED_MM).contains(&height_mm)
                {
                    return Err(format!(
                        "Custom bed dimensions ({width_mm} × {height_mm} mm) must lie within [{MIN_FABRICATION_BED_MM}, {MAX_FABRICATION_BED_MM}] mm"
                    ));
                }
                Ok((width_mm, height_mm))
            }
        }
    }

    /// Returns `(width_mm, height_mm)` clamped to `[20.0, 1200.0]` mm for live UI previews.
    #[must_use]
    pub fn clamped_dimensions_mm(&self) -> (f64, f64) {
        match *self {
            Self::A4 => (210.0, 297.0),
            Self::UsLetter => (215.9, 279.4),
            Self::Custom {
                width_mm,
                height_mm,
            } => {
                let w = if width_mm.is_finite() {
                    width_mm.clamp(MIN_FABRICATION_BED_MM, MAX_FABRICATION_BED_MM)
                } else {
                    200.0
                };
                let h = if height_mm.is_finite() {
                    height_mm.clamp(MIN_FABRICATION_BED_MM, MAX_FABRICATION_BED_MM)
                } else {
                    200.0
                };
                (w, h)
            }
        }
    }
}

/// Ordered 2D polyline or closed polygon loop for fabrication stroke serialization.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FabricationPolyline {
    /// Ordered `[x, y]` vertices.
    pub points: Vec<[f64; 2]>,
    /// When `true`, appends SVG `Z` to close the loop cleanly.
    pub closed: bool,
}

impl FabricationPolyline {
    /// Constructs a new fabrication polyline.
    #[must_use]
    pub fn new(points: Vec<[f64; 2]>, closed: bool) -> Self {
        Self { points, closed }
    }
}

/// Physical bed, margin, and stroke configuration for laser-cutter and pen-plotter SVG export.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FabricationSvgOptions {
    /// Target physical stock / bed size.
    pub bed: FabricationBedPreset,
    /// Uniform inset margin from all four sheet edges in millimeters.
    pub margin_mm: f64,
    /// CSS hex stroke color for `<g id="layer-cut">` (default `"#ff0000"`).
    pub cut_stroke_hex: String,
    /// CSS hex stroke color for `<g id="layer-score">` (default `"#0000ff"`).
    pub score_stroke_hex: String,
    /// Hairline stroke width in millimeters for `<g id="layer-cut">` (default `0.1`).
    pub cut_stroke_width_mm: f64,
    /// Pen / score stroke width in millimeters for `<g id="layer-score">` (default `0.15`).
    pub score_stroke_width_mm: f64,
    /// Maps Cartesian `+Y` upward coordinates into top-down SVG `+Y` downward millimeters.
    pub invert_y: bool,
}

impl Default for FabricationSvgOptions {
    fn default() -> Self {
        Self {
            bed: FabricationBedPreset::A4,
            margin_mm: 12.0,
            cut_stroke_hex: "#ff0000".to_string(),
            score_stroke_hex: "#0000ff".to_string(),
            cut_stroke_width_mm: 0.1,
            score_stroke_width_mm: 0.15,
            invert_y: true,
        }
    }
}

/// Aspect-preserving projection from mathematical `[min_x, max_x, min_y, max_y]` domain space
/// into physical `[0, width_mm] × [0, height_mm]` bed coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FabricationBedTransform {
    /// Total physical bed width in millimeters.
    pub width_mm: f64,
    /// Total physical bed height in millimeters.
    pub height_mm: f64,
    /// Safe inset margin in millimeters.
    pub margin_mm: f64,
    domain_min_x: f64,
    domain_min_y: f64,
    domain_span_y: f64,
    scale_mm_per_unit: f64,
    offset_x_mm: f64,
    offset_y_mm: f64,
    invert_y: bool,
}

impl FabricationBedTransform {
    /// Computes an aspect-preserving bed transform for `domain_bounds = [min_x, max_x, min_y, max_y]`.
    pub fn from_domain_bounds(
        domain_bounds: [f64; 4],
        options: &FabricationSvgOptions,
    ) -> Result<Self, String> {
        let (width_mm, height_mm) = options.bed.dimensions_mm()?;
        if !options.margin_mm.is_finite() || options.margin_mm < 0.0 {
            return Err("Fabrication margin_mm must be finite and non-negative".to_string());
        }
        let inner_w = width_mm - 2.0 * options.margin_mm;
        let inner_h = height_mm - 2.0 * options.margin_mm;
        if inner_w <= 0.0 || inner_h <= 0.0 {
            return Err(format!(
                "Margin {:.2} mm leaves zero drawable area on {:.1} × {:.1} mm bed",
                options.margin_mm, width_mm, height_mm
            ));
        }
        if domain_bounds.iter().any(|v| !v.is_finite()) {
            return Err("Domain bounds must be finite".to_string());
        }
        let span_x = (domain_bounds[1] - domain_bounds[0]).abs().max(1e-9);
        let span_y = (domain_bounds[3] - domain_bounds[2]).abs().max(1e-9);
        let scale = (inner_w / span_x).min(inner_h / span_y);
        let drawn_w = span_x * scale;
        let drawn_h = span_y * scale;
        Ok(Self {
            width_mm,
            height_mm,
            margin_mm: options.margin_mm,
            domain_min_x: domain_bounds[0].min(domain_bounds[1]),
            domain_min_y: domain_bounds[2].min(domain_bounds[3]),
            domain_span_y: span_y,
            scale_mm_per_unit: scale,
            offset_x_mm: options.margin_mm + 0.5 * (inner_w - drawn_w),
            offset_y_mm: options.margin_mm + 0.5 * (inner_h - drawn_h),
            invert_y: options.invert_y,
        })
    }

    /// Maps a single mathematical `[x, y]` point into physical `[x_mm, y_mm]` coordinates on the bed.
    #[must_use]
    pub fn project_point(&self, pt: [f64; 2]) -> [f64; 2] {
        let x_mm = self.offset_x_mm + (pt[0] - self.domain_min_x) * self.scale_mm_per_unit;
        let rel_y = pt[1] - self.domain_min_y;
        let y_mm = if self.invert_y {
            self.offset_y_mm + (self.domain_span_y - rel_y) * self.scale_mm_per_unit
        } else {
            self.offset_y_mm + rel_y * self.scale_mm_per_unit
        };
        [x_mm, y_mm]
    }

    /// Projects a slice of mathematical polylines into physical millimeter polylines.
    #[must_use]
    pub fn project_polylines(&self, paths: &[FabricationPolyline]) -> Vec<FabricationPolyline> {
        paths
            .iter()
            .filter_map(|path| {
                let pts: Vec<[f64; 2]> = path
                    .points
                    .iter()
                    .copied()
                    .filter(|p| p[0].is_finite() && p[1].is_finite())
                    .map(|p| self.project_point(p))
                    .collect();
                (pts.len() >= 2).then_some(FabricationPolyline {
                    points: pts,
                    closed: path.closed,
                })
            })
            .collect()
    }
}

/// Projects mathematical-domain `cut_paths` and `score_paths` onto the physical millimeter bed and
/// serializes a pure-vector fabrication SVG document.
pub fn export_domain_fabrication_svg(
    domain_bounds: [f64; 4],
    cut_paths: &[FabricationPolyline],
    score_paths: &[FabricationPolyline],
    options: &FabricationSvgOptions,
) -> Result<String, String> {
    let transform = FabricationBedTransform::from_domain_bounds(domain_bounds, options)?;
    let cut_mm = transform.project_polylines(cut_paths);
    let score_mm = transform.project_polylines(score_paths);
    export_fabrication_svg(&cut_mm, &score_mm, options)
}

/// Serializes physical millimeter `cut_paths_mm` (`#layer-cut`) and `score_paths_mm` (`#layer-score`)
/// into a laser-cutter / pen-plotter ready SVG document with zero `<rect>` or `<image>` elements.
pub fn export_fabrication_svg(
    cut_paths_mm: &[FabricationPolyline],
    score_paths_mm: &[FabricationPolyline],
    options: &FabricationSvgOptions,
) -> Result<String, String> {
    let (width_mm, height_mm) = options.bed.dimensions_mm()?;
    if !options.margin_mm.is_finite()
        || options.margin_mm < 0.0
        || 2.0 * options.margin_mm >= width_mm.min(height_mm)
    {
        return Err(format!(
            "Invalid fabrication margin {:.2} mm for {:.1} × {:.1} mm bed",
            options.margin_mm, width_mm, height_mm
        ));
    }

    let w_str = format_mm_header_value(width_mm);
    let h_str = format_mm_header_value(height_mm);
    let cut_stroke = sanitize_color_token(&options.cut_stroke_hex, "#ff0000");
    let score_stroke = sanitize_color_token(&options.score_stroke_hex, "#0000ff");
    let cut_sw = format_mm_header_value(options.cut_stroke_width_mm.clamp(0.01, 5.0));
    let score_sw = format_mm_header_value(options.score_stroke_width_mm.clamp(0.01, 5.0));

    let mut svg = String::with_capacity(512 + (cut_paths_mm.len() + score_paths_mm.len()) * 96);
    svg.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    svg.push_str(&format!(
        "<svg width=\"{w_str}mm\" height=\"{h_str}mm\" viewBox=\"0 0 {w_str} {h_str}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    ));

    write_fabrication_layer(
        &mut svg,
        "layer-score",
        score_paths_mm,
        &score_stroke,
        &score_sw,
    );
    write_fabrication_layer(&mut svg, "layer-cut", cut_paths_mm, &cut_stroke, &cut_sw);

    svg.push_str("</svg>\n");
    Ok(svg)
}

fn write_fabrication_layer(
    svg: &mut String,
    layer_id: &str,
    paths: &[FabricationPolyline],
    stroke: &str,
    stroke_width_mm: &str,
) {
    svg.push_str(&format!(
        "  <g id=\"{layer_id}\" fill=\"none\" stroke=\"{stroke}\" stroke-width=\"{stroke_width_mm}mm\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n"
    ));
    for path in paths {
        if let Some(d_attr) = build_svg_path_data(&path.points, path.closed) {
            svg.push_str(&format!(
                "    <path fill=\"none\" stroke=\"{stroke}\" stroke-width=\"{stroke_width_mm}mm\" d=\"{d_attr}\"/>\n"
            ));
        }
    }
    svg.push_str("  </g>\n");
}

fn build_svg_path_data(points: &[[f64; 2]], closed: bool) -> Option<String> {
    let mut valid: Vec<[f64; 2]> = points
        .iter()
        .copied()
        .filter(|p| p[0].is_finite() && p[1].is_finite())
        .collect();
    if valid.len() < 2 {
        return None;
    }
    if closed && valid.len() >= 3 {
        let first = valid[0];
        if let Some(&last) = valid.last() {
            let dx = last[0] - first[0];
            let dy = last[1] - first[1];
            if dx * dx + dy * dy <= 1e-12 {
                valid.pop();
            }
        }
    }
    if valid.len() < 2 {
        return None;
    }

    let mut d = String::with_capacity(valid.len() * 18);
    d.push_str(&format!("M {:.3} {:.3}", valid[0][0], valid[0][1]));
    for pt in &valid[1..] {
        d.push_str(&format!(" L {:.3} {:.3}", pt[0], pt[1]));
    }
    if closed {
        d.push_str(" Z");
    }
    Some(d)
}

fn format_mm_header_value(val: f64) -> String {
    let formatted = format!("{val:.4}");
    let trimmed = formatted.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "-0" {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

fn sanitize_color_token(raw: &str, fallback: &str) -> String {
    let trimmed = raw.trim();
    if !trimmed.is_empty()
        && trimmed.len() <= 32
        && trimmed
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '#' | '(' | ')' | ',' | '.' | '%'))
    {
        trimmed.to_string()
    } else {
        fallback.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_presets_and_custom_bounds_emit_exact_mm_headers_without_rect_or_image() {
        let cut = vec![FabricationPolyline::new(
            vec![[20.0, 20.0], [180.0, 20.0], [180.0, 180.0], [20.0, 180.0]],
            true,
        )];
        let score = vec![FabricationPolyline::new(
            vec![[30.0, 30.0], [170.0, 170.0]],
            false,
        )];

        let a4_svg =
            export_fabrication_svg(&cut, &score, &FabricationSvgOptions::default()).unwrap();
        assert!(a4_svg.contains("width=\"210mm\" height=\"297mm\" viewBox=\"0 0 210 297\""));
        assert!(!a4_svg.contains("<rect"));
        assert!(!a4_svg.contains("<image"));
        assert!(a4_svg.contains("<g id=\"layer-cut\""));
        assert!(a4_svg.contains("<g id=\"layer-score\""));
        assert!(a4_svg.contains("fill=\"none\""));
        assert!(a4_svg.contains(" Z\"/>"));

        let letter_opts = FabricationSvgOptions {
            bed: FabricationBedPreset::UsLetter,
            ..FabricationSvgOptions::default()
        };
        let letter_svg = export_fabrication_svg(&cut, &score, &letter_opts).unwrap();
        assert!(
            letter_svg.contains("width=\"215.9mm\" height=\"279.4mm\" viewBox=\"0 0 215.9 279.4\"")
        );

        let custom_opts = FabricationSvgOptions {
            bed: FabricationBedPreset::Custom {
                width_mm: 600.0,
                height_mm: 400.0,
            },
            margin_mm: 15.0,
            ..FabricationSvgOptions::default()
        };
        let custom_svg = export_fabrication_svg(&cut, &score, &custom_opts).unwrap();
        assert!(custom_svg.contains("width=\"600mm\" height=\"400mm\" viewBox=\"0 0 600 400\""));

        for invalid in [
            FabricationBedPreset::Custom {
                width_mm: 19.9,
                height_mm: 100.0,
            },
            FabricationBedPreset::Custom {
                width_mm: 100.0,
                height_mm: 1200.1,
            },
        ] {
            let bad = FabricationSvgOptions {
                bed: invalid,
                ..FabricationSvgOptions::default()
            };
            assert!(export_fabrication_svg(&cut, &score, &bad).is_err());
        }
    }

    #[test]
    fn test_domain_projection_respects_margins_and_aspect_ratio() {
        let opts = FabricationSvgOptions {
            bed: FabricationBedPreset::Custom {
                width_mm: 200.0,
                height_mm: 100.0,
            },
            margin_mm: 10.0,
            ..FabricationSvgOptions::default()
        };
        let tf = FabricationBedTransform::from_domain_bounds([0.0, 10.0, 0.0, 10.0], &opts)
            .expect("valid transform");
        // Drawable box is 180 x 80 mm -> square 10x10 domain scales to 80 x 80 mm, centered horizontally at x in [60, 140], y in [10, 90].
        let p_min = tf.project_point([0.0, 0.0]);
        let p_max = tf.project_point([10.0, 10.0]);
        assert!((p_min[0] - 60.0).abs() < 1e-6);
        assert!((p_min[1] - 90.0).abs() < 1e-6);
        assert!((p_max[0] - 140.0).abs() < 1e-6);
        assert!((p_max[1] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_exported_fabrication_svg_is_well_formed_xml() {
        let cut = vec![FabricationPolyline::new(
            vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            true,
        )];
        let score = vec![FabricationPolyline::new(
            vec![[1.0, 2.0], [5.0, 8.0], [9.0, 3.0]],
            false,
        )];
        let svg = export_domain_fabrication_svg(
            [0.0, 10.0, 0.0, 10.0],
            &cut,
            &score,
            &FabricationSvgOptions::default(),
        )
        .expect("valid SVG");
        assert!(svg.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg "));
        assert!(svg.trim_end().ends_with("</svg>"));
        assert_eq!(svg.matches("<g ").count(), 2);
        assert_eq!(svg.matches("</g>").count(), 2);
        assert_eq!(svg.matches("<path ").count(), 2);
        for line in svg.lines().skip(1) {
            let trimmed = line.trim();
            assert!(trimmed.starts_with('<') && trimmed.ends_with('>'));
        }
    }
}
