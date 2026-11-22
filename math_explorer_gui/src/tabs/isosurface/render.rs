//! Depth-sorted shaded 3D mesh projection onto [`crate::camera::Plot3DViewport`].
//!
//! Consolidates all 3D rotation and depth ordering through [`crate::camera::Camera3D::rotate_to_camera_space`]
//! without duplicating camera trigonometry, and shades facets using WCAG-compliant [`crate::app::SemanticPalette`]
//! colors across Dark, Light, and HighContrast themes.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::app::SemanticPalette;
use crate::camera::Camera3D;
use eframe::egui::{Color32, Stroke};
use egui_plot::{Line, PlotPoints, PlotUi, Polygon};
use oxidize_core::mesh::{Mesh, Point3D};

/// Upper bound on interactive viewport polygons drawn per frame so `64^3` TPMS
/// meshes remain responsive while `.stl` and `.obj` exports preserve 100% of triangles.
const MAX_VIEWPORT_POLYGONS: usize = 9_000;

/// Threshold above which occluded back-facing facets are culled prior to depth sorting.
const BACKFACE_CULL_THRESHOLD: usize = 6_000;

/// Projected, shaded 2D triangle ready for back-to-front painter emission.
#[derive(Clone, Copy, Debug)]
pub struct ProjectedFacet {
    /// Screen-space triangle vertices in `egui_plot` coordinates.
    pub points: [[f64; 2]; 3],
    /// Camera-space centroid depth (`viewer_z`, larger = closer to viewer).
    pub viewer_z: f64,
    /// Lit facet fill color.
    pub fill: Color32,
    /// Seam-sealing or wireframe edge stroke.
    pub stroke: Stroke,
}

const BOX_EDGES: [([f64; 3], [f64; 3]); 12] = [
    ([-1.0, -1.0, -1.0], [1.0, -1.0, -1.0]),
    ([1.0, -1.0, -1.0], [1.0, 1.0, -1.0]),
    ([1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]),
    ([-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]),
    ([-1.0, -1.0, 1.0], [1.0, -1.0, 1.0]),
    ([1.0, -1.0, 1.0], [1.0, 1.0, 1.0]),
    ([1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]),
    ([-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0]),
    ([-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0]),
    ([1.0, -1.0, -1.0], [1.0, -1.0, 1.0]),
    ([1.0, 1.0, -1.0], [1.0, 1.0, 1.0]),
    ([-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0]),
];

#[inline]
fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (f32::from(a) + (f32::from(b) - f32::from(a)) * t.clamp(0.0, 1.0)).round() as u8
}

#[inline]
fn blend_color(a: Color32, b: Color32, t: f32) -> Color32 {
    Color32::from_rgb(
        lerp_u8(a.r(), b.r(), t),
        lerp_u8(a.g(), b.g(), t),
        lerp_u8(a.b(), b.b(), t),
    )
}

fn compute_outward_normal(mesh: &Mesh, idx: [usize; 3]) -> Point3D {
    let v0 = mesh.vertices[idx[0]];
    let v1 = mesh.vertices[idx[1]];
    let v2 = mesh.vertices[idx[2]];
    let mut n = (v1 - v0).cross(v2 - v0).normalize();
    if mesh.normals.len() == mesh.vertices.len() {
        let grad_sum = mesh.normals[idx[0]] + mesh.normals[idx[1]] + mesh.normals[idx[2]];
        if grad_sum.magnitude() > Point3D::EPSILON {
            if n.magnitude() <= Point3D::EPSILON {
                n = grad_sum.normalize();
            } else if n.dot(grad_sum) < 0.0 {
                n = -n;
            }
        }
    }
    n
}

fn shade_facet(
    n_cam: [f64; 3],
    palette: &SemanticPalette,
    dark_mode: bool,
    show_wireframe: bool,
) -> (Color32, Stroke) {
    // Normalized studio key light [0.38, 0.62, 0.69] in camera space
    let key_dot = (n_cam[0] * 0.38 + n_cam[1] * 0.62 + n_cam[2] * 0.69) as f32;
    let front_facing = n_cam[2] >= 0.0;
    let base = if front_facing {
        palette.series[0]
    } else {
        palette.series[4]
    };

    let diffuse = key_dot.abs().clamp(0.0, 1.0);
    let rim = (1.0 - (n_cam[2].abs() as f32).clamp(0.0, 1.0)).powi(2) * 0.18;
    let lit_color = if dark_mode {
        let shadow = blend_color(palette.surface_fill, base, 0.48);
        let highlight = blend_color(base, palette.node_stroke, 0.42);
        blend_color(
            shadow,
            highlight,
            (0.22 + 0.68 * diffuse + rim).clamp(0.0, 1.0),
        )
    } else {
        let shadow = blend_color(base, palette.node_stroke, 0.38);
        let highlight = blend_color(base, palette.surface_fill, 0.35);
        blend_color(
            shadow,
            highlight,
            (0.20 + 0.72 * diffuse + rim).clamp(0.0, 1.0),
        )
    };

    let stroke = if show_wireframe {
        Stroke::new(0.75_f32, palette.node_stroke)
    } else {
        Stroke::new(0.45_f32, lit_color)
    };
    (lit_color, stroke)
}

/// Projects and depth-sorts `mesh` triangles into camera space using `Camera3D`.
#[must_use]
pub fn project_mesh_facets(
    mesh: &Mesh,
    camera: &Camera3D,
    palette: &SemanticPalette,
    dark_mode: bool,
    show_wireframe: bool,
) -> Vec<ProjectedFacet> {
    let total_tris = mesh.indices.len() / 3;
    if total_tris == 0 {
        return Vec::new();
    }
    let cull_backfaces = total_tris > BACKFACE_CULL_THRESHOLD;
    let zoom = f64::from(camera.zoom);
    let mut candidate_facets = Vec::with_capacity(total_tris.min(MAX_VIEWPORT_POLYGONS * 2));

    let raw_stride = if cull_backfaces {
        (total_tris / (MAX_VIEWPORT_POLYGONS * 2)).max(1)
    } else {
        1
    };

    for (tri_idx, &[i0, i1, i2]) in mesh.indices.as_chunks::<3>().0.iter().enumerate() {
        if raw_stride > 1 && !tri_idx.is_multiple_of(raw_stride) {
            continue;
        }
        if i0 >= mesh.vertices.len() || i1 >= mesh.vertices.len() || i2 >= mesh.vertices.len() {
            continue;
        }
        let n_world = compute_outward_normal(mesh, [i0, i1, i2]);
        let n_cam = camera.rotate_to_camera_space(&[
            f64::from(n_world.x),
            f64::from(n_world.y),
            f64::from(n_world.z),
        ]);
        if cull_backfaces && n_cam[2] < -0.15 {
            continue;
        }

        let mut pts = [[0.0_f64; 2]; 3];
        let mut z_sum = 0.0_f64;
        for (slot, &vi) in pts.iter_mut().zip(&[i0, i1, i2]) {
            let v = mesh.vertices[vi];
            let [rx, ry, rz] =
                camera.rotate_to_camera_space(&[f64::from(v.x), f64::from(v.y), f64::from(v.z)]);
            *slot = [rx * zoom, ry * zoom];
            z_sum += rz;
        }
        let (fill, stroke) = shade_facet(n_cam, palette, dark_mode, show_wireframe);
        candidate_facets.push(ProjectedFacet {
            points: pts,
            viewer_z: z_sum / 3.0,
            fill,
            stroke,
        });
    }

    if candidate_facets.len() > MAX_VIEWPORT_POLYGONS {
        let keep_stride = candidate_facets.len().div_ceil(MAX_VIEWPORT_POLYGONS);
        candidate_facets = candidate_facets
            .into_iter()
            .enumerate()
            .filter_map(|(i, f)| i.is_multiple_of(keep_stride).then_some(f))
            .collect();
    }

    candidate_facets.sort_by(|a, b| a.viewer_z.total_cmp(&b.viewer_z));
    candidate_facets
}

/// Draws the cubic domain bounding box and depth-sorted shaded mesh facets into `plot_ui`.
pub fn draw_isosurface_scene(
    plot_ui: &mut PlotUi<'_>,
    facets: &[ProjectedFacet],
    half_extent: f64,
    camera: &Camera3D,
    palette: &SemanticPalette,
) {
    for (edge_idx, (a, b)) in BOX_EDGES.iter().enumerate() {
        let p0 = camera.project(&[a[0] * half_extent, a[1] * half_extent, a[2] * half_extent]);
        let p1 = camera.project(&[b[0] * half_extent, b[1] * half_extent, b[2] * half_extent]);
        let name = if edge_idx == 0 { "domain_bounds" } else { "" };
        plot_ui.line(
            Line::new(name, PlotPoints::new(vec![p0, p1]))
                .color(palette.graph_edge)
                .width(1.0_f32)
                .allow_hover(false),
        );
    }

    for (idx, facet) in facets.iter().enumerate() {
        let name = if idx == facets.len() - 1 {
            "isosurface_mesh"
        } else {
            ""
        };
        plot_ui.polygon(
            Polygon::new(name, PlotPoints::from(facet.points.to_vec()))
                .fill_color(facet.fill)
                .stroke(facet.stroke)
                .allow_hover(false),
        );
    }
}
