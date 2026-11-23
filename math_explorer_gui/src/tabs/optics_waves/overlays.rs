#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::runner::OpticsFrameData;
use super::tool::OpticsWaveTool;
use crate::app::palette::SemanticPalette;
use crate::app::ThemeMode;
use eframe::egui::{self, pos2, Color32, Pos2, Rect, Stroke};
use math_explorer::physics::optics::refraction::{OpticalMediumConfig, OpticsPreset};

/// Theme-aware high-contrast colors for geometric ray, caustic, medium, and detector overlays.
#[derive(Clone, Copy, Debug)]
pub struct OpticsOverlayPalette {
    /// Contrasting halo stroke drawn beneath foreground lines.
    pub halo: Color32,
    /// Geometric ray trajectory stroke color.
    pub ray: Color32,
    /// Caustic cusp / focus highlight color.
    pub caustic: Color32,
    /// Lens / slab / aperture boundary outline color.
    pub boundary: Color32,
    /// Virtual detector line and irradiance profile color.
    pub detector: Color32,
}

/// Returns WCAG 2.2 AA (`>= 3.0:1`) contrast-verified overlay colors for `theme` via [`SemanticPalette`].
#[must_use]
pub fn optics_overlay_palette(theme: ThemeMode) -> OpticsOverlayPalette {
    let sem = SemanticPalette::for_theme(theme);
    let halo = match theme {
        ThemeMode::Light => Color32::WHITE,
        ThemeMode::Dark | ThemeMode::HighContrast => Color32::BLACK,
    };
    OpticsOverlayPalette {
        halo,
        ray: sem.accent,
        caustic: sem.node_stroke,
        boundary: sem.info,
        detector: sem.success,
    }
}

impl OpticsWaveTool {
    pub(super) fn paint_overlays(&self, ui: &egui::Ui, rect: Rect) {
        let theme = ThemeMode::from_visuals(ui.visuals());
        let pal = optics_overlay_palette(theme);
        let painter = ui.painter_at(rect);
        let (show_rays, det_x, fallback_medium) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.show_rays, p.detector_x, p.to_medium())
        };
        let medium = self
            .latest_frame
            .as_ref()
            .map_or(fallback_medium, |f| f.medium);

        Self::paint_medium_outline(&painter, rect, &medium, &pal);
        if let (true, Some(frame)) = (show_rays, self.latest_frame.as_ref()) {
            Self::paint_rays_and_caustics(&painter, rect, frame, &pal);
        }
        self.paint_detector_line(&painter, rect, det_x, &pal);
        painter.rect_stroke(
            rect,
            0.0,
            ui.visuals().widgets.noninteractive.bg_stroke,
            egui::StrokeKind::Inside,
        );
    }

    fn paint_medium_outline(
        painter: &egui::Painter,
        rect: Rect,
        medium: &OpticalMediumConfig,
        pal: &OpticsOverlayPalette,
    ) {
        let to_screen = |x: f64, y: f64| -> Pos2 {
            pos2(
                rect.left() + (x as f32) * rect.width(),
                rect.top() + (y as f32) * rect.height(),
            )
        };
        match medium.preset {
            OpticsPreset::LuneburgLens
            | OpticsPreset::MaxwellFishEye
            | OpticsPreset::BiconvexDroplet => {
                let [cx, cy] = medium.lens_center();
                let center = to_screen(cx, cy);
                let r_px = (medium.lens_radius as f32) * rect.width();
                painter.circle_stroke(center, r_px, Stroke::new(3.0_f32, pal.halo));
                painter.circle_stroke(center, r_px, Stroke::new(1.6_f32, pal.boundary));
            }
            OpticsPreset::VeselagoLens => {
                let [xa, xb] = medium.veselago_slab_bounds();
                let slab = Rect::from_min_max(to_screen(xa, 0.08), to_screen(xb, 0.92));
                painter.rect_stroke(
                    slab,
                    0.0,
                    Stroke::new(3.0_f32, pal.halo),
                    egui::StrokeKind::Middle,
                );
                painter.rect_stroke(
                    slab,
                    0.0,
                    Stroke::new(1.6_f32, pal.boundary),
                    egui::StrokeKind::Middle,
                );
            }
            OpticsPreset::DoubleSlit => {
                let wx = medium.slit_wall_x();
                painter.line_segment(
                    [to_screen(wx, 0.04), to_screen(wx, 0.96)],
                    Stroke::new(1.2_f32, pal.boundary),
                );
            }
        }
    }

    fn paint_rays_and_caustics(
        painter: &egui::Painter,
        rect: Rect,
        frame: &OpticsFrameData,
        pal: &OpticsOverlayPalette,
    ) {
        let to_screen = |p: [f64; 2]| -> Pos2 {
            pos2(
                rect.left() + (p[0] as f32) * rect.width(),
                rect.top() + (p[1] as f32) * rect.height(),
            )
        };
        for ray in &frame.rays {
            if ray.len() >= 2 {
                let pts: Vec<Pos2> = ray.iter().copied().map(to_screen).collect();
                painter.add(egui::Shape::line(
                    pts.clone(),
                    Stroke::new(2.4_f32, pal.halo),
                ));
                painter.add(egui::Shape::line(pts, Stroke::new(1.2_f32, pal.ray)));
            }
        }
        for &cp in &frame.caustics {
            let pt = to_screen(cp);
            painter.circle_filled(pt, 3.6_f32, pal.halo);
            painter.circle_filled(pt, 2.2_f32, pal.caustic);
        }
    }

    fn paint_detector_line(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        det_x: f64,
        pal: &OpticsOverlayPalette,
    ) {
        let x_px = rect.left() + (det_x as f32) * rect.width();
        let top = pos2(x_px, rect.top() + 0.08 * rect.height());
        let bot = pos2(x_px, rect.top() + 0.92 * rect.height());
        painter.line_segment([top, bot], Stroke::new(3.4_f32, pal.halo));
        painter.line_segment([top, bot], Stroke::new(1.8_f32, pal.detector));

        if let Some(ref frame) = self.latest_frame {
            let prof = &frame.detector_profile;
            let peak = frame.readout.peak_irradiance.max(1e-9);
            if prof.len() >= 8 {
                let max_bar_w = 0.08 * rect.width();
                let mut curve = Vec::with_capacity(prof.len());
                for (i, &val) in prof.iter().enumerate() {
                    let y_frac = (i as f32 + 0.5) / (prof.len() as f32);
                    let amp = ((val / peak).clamp(0.0, 1.0) as f32) * max_bar_w;
                    curve.push(pos2(
                        (x_px + amp).min(rect.right() - 2.0),
                        rect.top() + y_frac * rect.height(),
                    ));
                }
                painter.add(egui::Shape::line(
                    curve.clone(),
                    Stroke::new(2.6_f32, pal.halo),
                ));
                painter.add(egui::Shape::line(curve, Stroke::new(1.4_f32, pal.detector)));
            }
        }
    }
}
