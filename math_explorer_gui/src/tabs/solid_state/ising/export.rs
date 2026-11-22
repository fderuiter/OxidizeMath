//! Discrete domain-wall `<path>` SVG and toroidal 16-bit texture exporters for [`super::IsingModelTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{ising_spin_color, IsingModelTool, DEFAULT_ISING_SIZE};
use eframe::egui;
use egui_plot::export::ExportOptions;
use egui_plot::export_primitives::{export_contour_layers_svg, SvgContourLayer, SvgContourPath};
use math_commons::math_kernel::marching_squares::{MarchingSquares, ScalarField2D};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::stat_mech::ising::SpinLattice;
use math_explorer::physics::stat_mech::KB;

impl IsingModelTool {
    fn resolve_spins(&self) -> Option<(usize, usize, Vec<i8>)> {
        if let Some((w, h, ref spins)) = self.stepped_spins {
            if w > 0 && h > 0 && spins.len() == w * h {
                return Some((w, h, spins.clone()));
            }
        }
        if let Some(snap) = self.worker.latest_snapshot() {
            if let Some(spins) = snap
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<Vec<i8>>())
                .filter(|s| s.len() == snap.width * snap.height && snap.width > 0)
            {
                return Some((snap.width, snap.height, spins.clone()));
            }
        }
        let lattice = SpinLattice::new(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, Some(42));
        Some((lattice.width(), lattice.height(), lattice.spins().to_vec()))
    }

    pub(super) fn build_ising_raster_image(&self) -> Option<egui::ColorImage> {
        if let Some((w, h, ref px)) = self.stepped_pixels {
            if w > 0 && h > 0 && !px.is_empty() {
                return Some(egui::ColorImage::new([w, h], px.clone()));
            }
        }
        let snap = self.worker.latest_snapshot()?;
        let px = snap.pixels.try_read().ok()?;
        if px.is_empty() || snap.width == 0 || snap.height == 0 {
            return None;
        }
        Some(egui::ColorImage::new([snap.width, snap.height], px.clone()))
    }

    pub(super) fn build_ising_scalar_field(&self) -> Option<ScalarField2D> {
        let (w, h, spins) = self.resolve_spins()?;
        let mut values = vec![0.0_f64; w * h];
        for y in 0..h {
            let yp = if y + 1 < h { y + 1 } else { 0 };
            let ym = if y > 0 { y - 1 } else { h - 1 };
            for x in 0..w {
                let xp = if x + 1 < w { x + 1 } else { 0 };
                let xm = if x > 0 { x - 1 } else { w - 1 };
                let center = f64::from(spins[flatten_2d_index(x, y, w)]);
                let neighbors = f64::from(spins[flatten_2d_index(xp, y, w)])
                    + f64::from(spins[flatten_2d_index(xm, y, w)])
                    + f64::from(spins[flatten_2d_index(x, yp, w)])
                    + f64::from(spins[flatten_2d_index(x, ym, w)]);
                values[flatten_2d_index(x, y, w)] = 0.6 * center + 0.1 * neighbors;
            }
        }
        Some(ScalarField2D {
            width: w,
            height: h,
            values,
            periodic: true,
            default_levels: 4,
        })
    }

    pub(super) fn build_ising_vector_svg(&self, options: &ExportOptions) -> Result<String, String> {
        let (w, h, spins) = self
            .resolve_spins()
            .ok_or_else(|| "Ising spin lattice unavailable for SVG export".to_string())?;
        let raw_field = ScalarField2D {
            width: w,
            height: h,
            values: spins.iter().map(|&s| f64::from(s)).collect(),
            periodic: true,
            default_levels: 1,
        };
        let extracted = MarchingSquares::extract_at_thresholds(&raw_field, &[0.0]);
        let bg_sum = u16::from(options.background_color.r())
            + u16::from(options.background_color.g())
            + u16::from(options.background_color.b());
        let stroke = if bg_sum < 384 {
            crate::app::ThemeMode::Dark.palette().axis_x
        } else {
            crate::app::ThemeMode::Light.palette().axis_x
        };
        let mut paths: Vec<SvgContourPath> = extracted
            .into_iter()
            .flat_map(|lvl| lvl.paths)
            .map(|p| SvgContourPath {
                points: p.points,
                closed: p.closed,
            })
            .collect();
        if paths.is_empty() {
            paths.push(SvgContourPath {
                points: vec![
                    [0.0, 0.0],
                    [w as f64, 0.0],
                    [w as f64, h as f64],
                    [0.0, h as f64],
                ],
                closed: true,
            });
        }
        let layer = SvgContourLayer {
            label: "ising-domain-walls".to_string(),
            color: stroke,
            stroke_width: 1.25,
            fill: None,
            paths,
        };
        export_contour_layers_svg(w as f64, h as f64, &[layer], options)
    }

    pub(super) fn step_ising_offscreen(&mut self) {
        let mut lattice = SpinLattice::new(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, None);
        let j_scale = if self.j_coupling.abs() < 1e-6 {
            1.0
        } else {
            self.j_coupling.abs()
        };
        let t_phys = self.temperature * j_scale / KB;
        lattice.evolve(
            self.steps_per_frame.clamp(100, 5000),
            t_phys,
            self.j_coupling,
            self.h_field,
        );
        let palette = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .theme
            .palette();
        let spins = lattice.spins().to_vec();
        let pixels: Vec<egui::Color32> = spins
            .iter()
            .map(|&s| ising_spin_color(s, &palette))
            .collect();
        self.stepped_pixels = Some((lattice.width(), lattice.height(), pixels));
        self.stepped_spins = Some((lattice.width(), lattice.height(), spins));
        self.dirty = true;
    }
}
