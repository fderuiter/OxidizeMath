//! Discrete domain-wall `<path>` SVG and toroidal 16-bit texture exporters for [`super::IsingModelTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::preset::IsingInteractionMode;
use super::{ising_spin_color, IsingModelTool, DEFAULT_ISING_SIZE};
use eframe::egui;
use egui_plot::export::ExportOptions;
use egui_plot::export_primitives::{export_contour_layers_svg, SvgContourLayer, SvgContourPath};
use math_commons::math_kernel::marching_squares::{MarchingSquares, ScalarField2D};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::applied::game_theory::evolutionary::MatrixPayoff;
use math_explorer::physics::stat_mech::ising::SpinLattice;
use math_explorer::physics::stat_mech::KB;
use nalgebra::DMatrix;

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
        if self.is_graph_mode() {
            return None;
        }
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
        if self.is_graph_mode() {
            return self.build_csr_graph_vector_svg(options);
        }
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

    fn build_csr_graph_vector_svg(&self, options: &ExportOptions) -> Result<String, String> {
        let snap = self
            .shared_topology
            .as_ref()
            .ok_or_else(|| "Shared CSR topology unavailable".to_string())?;
        let spins = self
            .graph_system
            .as_ref()
            .map(math_explorer::physics::stat_mech::GraphSpinSystem::spins)
            .ok_or_else(|| "CSR spin system unavailable".to_string())?;
        let bg_sum = u16::from(options.background_color.r())
            + u16::from(options.background_color.g())
            + u16::from(options.background_color.b());
        let pal = if bg_sum < 384 {
            crate::app::ThemeMode::Dark.palette()
        } else {
            crate::app::ThemeMode::Light.palette()
        };
        let mapped = snap.map_positions_to_bounds(24.0, 24.0, 232.0, 232.0);
        let edge_paths: Vec<SvgContourPath> = snap
            .undirected_edges
            .iter()
            .filter_map(|&(u, v, _)| {
                let (&pu, &pv) = (mapped.get(u)?, mapped.get(v)?);
                Some(SvgContourPath {
                    points: vec![
                        [f64::from(pu[0]), f64::from(pu[1])],
                        [f64::from(pv[0]), f64::from(pv[1])],
                    ],
                    closed: false,
                })
            })
            .collect();
        let (up_layer, down_layer) = csr_spin_node_layers(&mapped, spins, &pal);
        let edge_layer = SvgContourLayer {
            label: "csr-edges".into(),
            color: pal.graph_edge,
            stroke_width: 1.2,
            fill: None,
            paths: edge_paths,
        };
        export_contour_layers_svg(256.0, 256.0, &[edge_layer, up_layer, down_layer], options)
    }

    pub(super) fn step_ising_offscreen(&mut self) {
        if self.is_graph_mode() {
            let trials = self.steps_per_frame.clamp(16, 2_000);
            self.step_graph_trials(trials);
            return;
        }
        let mut lattice = SpinLattice::new(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, None);
        let steps = self.steps_per_frame.clamp(100, 5000);
        match self.mode {
            IsingInteractionMode::Ferromagnetic => {
                let t_phys = self.temperature.max(1e-3) / KB;
                lattice.evolve(steps, t_phys, self.j_coupling, self.h_field);
            }
            IsingInteractionMode::FermiGame => {
                let mat = DMatrix::from_row_slice(2, 2, &self.game_payoffs);
                if let Ok(payoff) = MatrixPayoff::new(mat) {
                    lattice.evolve_fermi_game_with_mutation(
                        steps,
                        self.temperature,
                        &payoff,
                        self.mutation_rate,
                    );
                }
            }
        }
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

fn csr_spin_node_layers(
    mapped: &[[f32; 2]],
    spins: &[i8],
    pal: &crate::app::SemanticPalette,
) -> (SvgContourLayer, SvgContourLayer) {
    let (mut up_nodes, mut down_nodes) = (Vec::new(), Vec::new());
    for (i, &[cx, cy]) in mapped.iter().enumerate() {
        let poly: Vec<[f64; 2]> = (0..8)
            .map(|k| {
                let th = (k as f64) * std::f64::consts::TAU / 8.0;
                [
                    f64::from(cx) + 4.5 * th.cos(),
                    f64::from(cy) + 4.5 * th.sin(),
                ]
            })
            .collect();
        let path = SvgContourPath {
            points: poly,
            closed: true,
        };
        if spins.get(i).copied().unwrap_or(1) > 0 {
            up_nodes.push(path);
        } else {
            down_nodes.push(path);
        }
    }
    (
        SvgContourLayer {
            label: "csr-up".into(),
            color: pal.axis_x,
            stroke_width: 1.0,
            fill: Some(pal.axis_x),
            paths: up_nodes,
        },
        SvgContourLayer {
            label: "csr-down".into(),
            color: pal.axis_z,
            stroke_width: 1.0,
            fill: Some(pal.axis_z),
            paths: down_nodes,
        },
    )
}
