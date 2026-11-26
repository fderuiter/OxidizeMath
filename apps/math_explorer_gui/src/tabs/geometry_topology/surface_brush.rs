#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Opt-in 3D front-facing raycast picking and WCAG 2.2 AA keyboard geodesic brush controller
//! for triangulated 2-manifolds rendered inside [`crate::plot3d_viewport::Plot3DViewport`].

use crate::accessibility::{announce_live_status, paint_canvas_focus_ring};
use crate::app::palette::SemanticPalette;
use crate::camera::Camera3D;
use eframe::egui;
use egui_plot::commands::{CommandRegistryData, CommandTrigger};
use egui_plot::{Points, Polygon};
use math_explorer::pure_math::differential_geometry::CotangentLaplacian;
use serde::{Deserialize, Serialize};

/// Helper caption displayed beneath [`crate::plot3d_viewport::Plot3DViewport`] in Sculpt mode.
pub const SCULPT_3D_HELPER_TEXT: &str =
    "Sculpt mode: Drag surface or use Arrow keys + Space/Enter to paint ([ / ] radius); Shift+Drag / Middle-Drag pans, Scroll / Q / E zooms.";

/// Primary pointer interaction mode on a 3D manifold viewport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ViewportBrushMode {
    /// Primary drag orbits [`Camera3D`]; keyboard uses standard camera shortcuts.
    #[default]
    Orbit,
    /// Primary drag and keyboard arrows paint geodesic perturbations onto front-facing 3D vertices.
    Sculpt,
}

/// Sign of the perturbation injected by a 3D geodesic brush stroke.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum BrushPolarity {
    /// Positive morphogen seed, heat pulse, or depolarizing current injection.
    #[default]
    Excite,
    /// Negative sink, cooling pulse, or hyperpolarizing inhibition (`Alt` modifier or toggle).
    Inhibit,
}

/// Result of front-facing 3D raycast picking in projected viewport coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceBrushHit {
    /// Picked vertex index in `0..positions.len()`.
    pub vertex: usize,
    /// 2D projected distance from pointer to vertex in plot units.
    pub distance_2d: f64,
    /// Camera-space depth (`viewer_z`, larger = closer to viewer).
    pub viewer_z: f64,
}

/// Discrete or continuous geodesic brush stroke emitted by pointer drag or keyboard action.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeodesicBrushStroke {
    /// Center vertex index of the stroke.
    pub center_vertex: usize,
    /// Spatial Gaussian falloff sigma in world units.
    pub radius: f64,
    /// Peak amplitude of the perturbation.
    pub strength: f64,
    /// Excitation vs inhibition polarity.
    pub polarity: BrushPolarity,
    /// Whether this stroke originated from keyboard `Space`/`Enter` (`true`) or pointer (`false`).
    pub via_keyboard: bool,
}

/// Stateful 3D surface brush controller managing raycast hover, pointer strokes, and geodesic cursor navigation.
#[derive(Debug, Clone)]
pub struct SurfaceBrushController {
    /// Current viewport gesture mode (`Orbit` vs `Sculpt`).
    pub mode: ViewportBrushMode,
    /// Current stroke polarity (`Excite` vs `Inhibit`).
    pub polarity: BrushPolarity,
    /// Geodesic kernel radius in world units (`0.05..=1.20`).
    pub radius: f64,
    /// Brush perturbation strength (`0.05..=2.0`).
    pub strength: f64,
    /// Active keyboard / hover geodesic cursor vertex index.
    pub cursor_vertex: usize,
    /// Most recent pointer raycast hit for visual hover feedback.
    pub hovered_hit: Option<SurfaceBrushHit>,
    announced_drag_start: bool,
}

impl Default for SurfaceBrushController {
    fn default() -> Self {
        Self {
            mode: ViewportBrushMode::Orbit,
            polarity: BrushPolarity::Excite,
            radius: 0.28,
            strength: 0.65,
            cursor_vertex: 0,
            hovered_hit: None,
            announced_drag_start: false,
        }
    }
}

pub use super::surface_brush_kernel::{apply_geodesic_kernel, pick_front_facing_vertex};

/// Borrowed manifold geometry slice bundle passed to [`SurfaceBrushController::handle_viewport_response`].
#[derive(Clone, Copy)]
pub struct ManifoldPickView<'a> {
    /// Active 3D camera pose.
    pub camera: &'a Camera3D,
    /// CSR cotangent operator providing 1-ring geodesic neighbors.
    pub laplacian: &'a CotangentLaplacian,
    /// World-space vertex coordinates.
    pub positions: &'a [[f64; 3]],
    /// Indexed triangle triples.
    pub triangles: &'a [[usize; 3]],
    /// Outward vertex normals.
    pub normals: &'a [[f64; 3]],
    /// Whether back-facing triangles are pickable (Klein / open patches).
    pub two_sided: bool,
}

impl SurfaceBrushController {
    /// Clamps `cursor_vertex` into `0..vertex_count` after a mesh resolution rebuild.
    pub fn clamp_cursor(&mut self, vertex_count: usize) {
        if vertex_count == 0 {
            self.cursor_vertex = 0;
        } else if self.cursor_vertex >= vertex_count {
            self.cursor_vertex = vertex_count - 1;
        }
    }

    /// Renders the high-contrast 3D geodesic cursor and hover reticle inside `PlotUi`.
    pub fn draw_plot_overlays(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        camera: &Camera3D,
        positions: &[[f64; 3]],
        palette: SemanticPalette,
    ) {
        if self.mode != ViewportBrushMode::Sculpt || positions.is_empty() {
            return;
        }
        let cursor_idx = self.cursor_vertex.min(positions.len() - 1);
        let cursor_2d = camera.project(&positions[cursor_idx]);
        let r_plot = (self.radius * f64::from(camera.zoom)).clamp(0.05, 1.5);
        let ring: Vec<[f64; 2]> = (0..24)
            .map(|k| {
                let ang = std::f64::consts::TAU * (k as f64) / 24.0;
                [
                    cursor_2d[0] + r_plot * ang.cos(),
                    cursor_2d[1] + r_plot * ang.sin(),
                ]
            })
            .collect();
        plot_ui.polygon(
            Polygon::new("Geodesic Brush Radius", ring)
                .fill_color(egui::Color32::TRANSPARENT)
                .stroke(egui::Stroke::new(2.0_f32, palette.accent)),
        );
        plot_ui.points(
            Points::new("Geodesic Cursor Vertex", vec![cursor_2d])
                .radius(5.5_f32)
                .color(palette.node_stroke),
        );
        if let Some(hit) = self.hovered_hit {
            if hit.vertex < positions.len() && hit.vertex != cursor_idx {
                let hp = camera.project(&positions[hit.vertex]);
                plot_ui.points(
                    Points::new("Hovered Manifold Vertex", vec![hp])
                        .radius(4.5_f32)
                        .color(palette.info),
                );
            }
        }
    }

    /// Samples pointer raycast drags and WCAG 2.2 AA keyboard geodesic navigation after `Plot3DViewport::show`.
    pub fn handle_viewport_response(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        pointer_plot_xy: Option<[f64; 2]>,
        view: ManifoldPickView<'_>,
    ) -> Option<GeodesicBrushStroke> {
        self.clamp_cursor(view.positions.len());
        paint_canvas_focus_ring(ui, response);
        if self.mode != ViewportBrushMode::Sculpt || view.positions.is_empty() {
            self.hovered_hit = None;
            self.announced_drag_start = false;
            return None;
        }
        register_sculpt_shortcuts(ui.ctx());

        self.hovered_hit = pointer_plot_xy.and_then(|xy| {
            pick_front_facing_vertex(
                view.camera,
                view.positions,
                view.triangles,
                view.normals,
                xy,
                view.two_sided,
                0.35,
            )
        });

        if let Some(stroke) = self.poll_pointer_stroke(ui, response) {
            return Some(stroke);
        }
        self.poll_keyboard_cursor(ui, response, view.camera, view.laplacian, view.positions)
    }

    fn poll_pointer_stroke(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
    ) -> Option<GeodesicBrushStroke> {
        let shift_held = ui.input(|i| i.modifiers.shift);
        if shift_held {
            return None;
        }
        let primary_active =
            response.clicked() || response.dragged_by(egui::PointerButton::Primary);
        if !primary_active {
            self.announced_drag_start = false;
            return None;
        }
        let hit = self.hovered_hit?;
        self.cursor_vertex = hit.vertex;
        let alt_held = ui.input(|i| i.modifiers.alt);
        let polarity = if alt_held {
            BrushPolarity::Inhibit
        } else {
            self.polarity
        };
        if !self.announced_drag_start {
            self.announced_drag_start = true;
            announce_live_status(
                ui.ctx(),
                format!("Sculpt stroke on manifold vertex {}", hit.vertex),
            );
        }
        Some(GeodesicBrushStroke {
            center_vertex: hit.vertex,
            radius: self.radius,
            strength: self.strength,
            polarity,
            via_keyboard: false,
        })
    }

    fn poll_keyboard_cursor(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        camera: &Camera3D,
        laplacian: &CotangentLaplacian,
        positions: &[[f64; 3]],
    ) -> Option<GeodesicBrushStroke> {
        if !response.has_focus() {
            return None;
        }
        lock_arrow_focus(ui, response.id);
        let cmd = read_keyboard_brush_command(ui);

        if cmd.radius_delta != 0.0 {
            self.radius = (self.radius + cmd.radius_delta).clamp(0.05, 1.20);
            announce_live_status(
                ui.ctx(),
                format!("Geodesic brush radius {:.2}", self.radius),
            );
        }
        if let Some(target_dir) = cmd.dir {
            for _ in 0..cmd.hops {
                self.cursor_vertex = step_geodesic_neighbor(
                    camera,
                    laplacian,
                    positions,
                    self.cursor_vertex,
                    target_dir,
                );
            }
            let p = positions[self.cursor_vertex];
            announce_live_status(
                ui.ctx(),
                format!(
                    "Geodesic cursor vertex {} at ({:.2}, {:.2}, {:.2})",
                    self.cursor_vertex, p[0], p[1], p[2]
                ),
            );
        }
        if !cmd.trigger_paint {
            return None;
        }
        let polarity = if cmd.alt_held {
            BrushPolarity::Inhibit
        } else {
            self.polarity
        };
        announce_live_status(
            ui.ctx(),
            format!("Applied geodesic brush at vertex {}", self.cursor_vertex),
        );
        Some(GeodesicBrushStroke {
            center_vertex: self.cursor_vertex,
            radius: self.radius,
            strength: self.strength,
            polarity,
            via_keyboard: true,
        })
    }
}

struct KeyboardBrushCommand {
    dir: Option<[f64; 2]>,
    hops: usize,
    radius_delta: f64,
    trigger_paint: bool,
    alt_held: bool,
}

fn register_sculpt_shortcuts(ctx: &egui::Context) {
    CommandRegistryData::register(
        ctx,
        "Arrow Keys",
        "Step geodesic cursor along manifold 1-ring",
        CommandTrigger::Key(egui::Key::ArrowRight),
        false,
        "Geodesic Brush",
    );
    CommandRegistryData::register(
        ctx,
        "[ / ]",
        "Decrease or increase geodesic brush radius",
        CommandTrigger::Key(egui::Key::OpenBracket),
        false,
        "Geodesic Brush",
    );
    CommandRegistryData::register(
        ctx,
        "Space / Enter",
        "Inject geodesic brush stroke at cursor vertex",
        CommandTrigger::Key(egui::Key::Space),
        false,
        "Geodesic Brush",
    );
}

fn lock_arrow_focus(ui: &egui::Ui, id: egui::Id) {
    ui.memory_mut(|m| {
        m.set_focus_lock_filter(
            id,
            egui::EventFilter {
                tab: false,
                horizontal_arrows: true,
                vertical_arrows: true,
                escape: false,
            },
        );
    });
}

fn read_keyboard_brush_command(ui: &egui::Ui) -> KeyboardBrushCommand {
    ui.input(|i| KeyboardBrushCommand {
        dir: if i.key_pressed(egui::Key::ArrowLeft) {
            Some([-1.0, 0.0])
        } else if i.key_pressed(egui::Key::ArrowRight) {
            Some([1.0, 0.0])
        } else if i.key_pressed(egui::Key::ArrowUp) {
            Some([0.0, 1.0])
        } else if i.key_pressed(egui::Key::ArrowDown) {
            Some([0.0, -1.0])
        } else {
            None
        },
        hops: if i.modifiers.shift { 3 } else { 1 },
        radius_delta: if i.key_pressed(egui::Key::OpenBracket) {
            -0.05
        } else if i.key_pressed(egui::Key::CloseBracket) {
            0.05
        } else {
            0.0
        },
        trigger_paint: i.key_pressed(egui::Key::Space) || i.key_pressed(egui::Key::Enter),
        alt_held: i.modifiers.alt,
    })
}

fn step_geodesic_neighbor(
    camera: &Camera3D,
    laplacian: &CotangentLaplacian,
    positions: &[[f64; 3]],
    current: usize,
    target_dir: [f64; 2],
) -> usize {
    let n = positions.len();
    if current >= n {
        return 0;
    }
    let p_curr = camera.project(&positions[current]);
    let mut best_nb = None;
    let mut best_align = 0.05_f64;
    for &nb in laplacian.neighbors(current) {
        if nb >= n {
            continue;
        }
        let p_nb = camera.project(&positions[nb]);
        let dx = p_nb[0] - p_curr[0];
        let dy = p_nb[1] - p_curr[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len > 1e-8 {
            let align = (dx * target_dir[0] + dy * target_dir[1]) / len;
            if align > best_align {
                best_align = align;
                best_nb = Some(nb);
            }
        }
    }
    best_nb.unwrap_or_else(|| (current + 1) % n)
}
