#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Discretized 3D manifold mesh builder (`<= 4,096` vertices) producing welded polar,
//! periodic, folded cortical, and compiled custom surfaces alongside [`CotangentLaplacian`]
//! and [`DiscreteSurface`].

use super::super::custom_surface::CustomSurfaceSandbox;
use super::state::{ManifoldKind, RiemannianStudioConfig, MAX_MANIFOLD_VERTICES};
use math_explorer::pure_math::differential_geometry::mean_curvature_flow::DiscreteSurface;
use math_explorer::pure_math::differential_geometry::surface::{
    KleinBottle, ParametricSurface, ParametricSurfaceKind, Sphere, SurfaceAnalysis, Torus,
};
use math_explorer::pure_math::differential_geometry::CotangentLaplacian;
use nalgebra::Point3;
use std::f64::consts::{PI, TAU};

/// Indexed 3D manifold geometry paired with its CSR cotangent Laplace-Beltrami operator.
#[derive(Clone)]
pub struct ManifoldGeometry {
    /// Vertex world coordinates `[x, y, z]` (`len <= 4,096`).
    pub positions: Vec<[f64; 3]>,
    /// Outward unit vertex normals `[nx, ny, nz]`.
    pub normals: Vec<[f64; 3]>,
    /// Outward-wound triangle index triples `[i0, i1, i2]`.
    pub triangles: Vec<[usize; 3]>,
    /// Parameter-chart coordinates `(u, v)` per vertex.
    pub uvs: Vec<(f64, f64)>,
    /// Mapping from vertex index to `(i, j)` in `discrete_surface`.
    pub vertex_grid_ij: Vec<(usize, usize)>,
    /// Structured grid dimensions `(nu, nv)`.
    pub grid_dims: (usize, usize),
    /// Parameter domain `((u_min, u_max), (v_min, v_max))`.
    pub uv_domain: ((f64, f64), (f64, f64)),
    /// Whether the mesh is open or non-orientable (enables two-sided picking and shell extrusion on export).
    pub two_sided: bool,
    /// Discrete mean curvature $|H|_i$ per vertex.
    pub mean_curvature: Vec<f64>,
    /// Analytical or discrete Gaussian curvature $K_i$ per vertex.
    pub gaussian_curvature: Vec<f64>,
    /// CSR cotangent Laplace-Beltrami operator.
    pub laplacian: CotangentLaplacian,
    /// Structured [`DiscreteSurface`] used by `MeanCurvatureFlow`.
    pub discrete_surface: DiscreteSurface,
}

fn normalize3(v: [f64; 3], fallback: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n.is_finite() && n > 1e-12 {
        [v[0] / n, v[1] / n, v[2] / n]
    } else {
        fallback
    }
}

/// Evaluates a folded gyral-sulcal cortical white-matter point and its radial pial thickness (mm).
#[must_use]
pub fn cortical_shell_point_and_thickness(
    u: f64,
    v: f64,
    sulcal_depth: f64,
) -> ([f64; 3], [f64; 3], f64) {
    let (sv, cv) = v.sin_cos();
    let (su, cu) = u.sin_cos();
    let fold = (5.0 * u).sin() * (4.0 * v).sin() * 0.55
        + (3.0 * u + 2.0 * v).cos() * (3.0 * v).sin() * 0.45;
    // Polar taper keeps north/south poles smooth and single-valued at v = 0 and v = PI.
    let polar_envelope = sv * sv;
    let disp = sulcal_depth * polar_envelope * fold;
    let r = (1.15 + disp).max(0.45);
    let dir = [sv * cu * 1.08, sv * su * 0.90, cv * 0.84];
    let white = [dir[0] * r, dir[1] * r, dir[2] * r];
    let crown_norm = (0.5 * (fold * polar_envelope + 1.0)).clamp(0.0, 1.0);
    // Gyral crowns (~3.2 mm) are thicker than sulcal fundi (~1.8 mm) per Fischl & Dale (2000).
    let thickness_mm = 1.80 + 1.40 * crown_norm;
    let world_offset = thickness_mm * 0.035;
    let radial = normalize3(dir, [0.0, 0.0, 1.0]);
    let pial = [
        white[0] + radial[0] * world_offset,
        white[1] + radial[1] * world_offset,
        white[2] + radial[2] * world_offset,
    ];
    (white, pial, thickness_mm)
}

/// Recomputes smooth area-weighted vertex normals directly from indexed `positions` and `triangles`.
pub fn recompute_vertex_normals(
    positions: &[[f64; 3]],
    triangles: &[[usize; 3]],
    out_normals: &mut [[f64; 3]],
) {
    for slot in out_normals.iter_mut() {
        *slot = [0.0, 0.0, 0.0];
    }
    for &[i0, i1, i2] in triangles {
        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }
        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];
        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        for idx in [i0, i1, i2] {
            out_normals[idx][0] += n[0];
            out_normals[idx][1] += n[1];
            out_normals[idx][2] += n[2];
        }
    }
    for (idx, slot) in out_normals.iter_mut().enumerate() {
        let fallback = normalize3(positions[idx], [0.0, 0.0, 1.0]);
        *slot = normalize3(*slot, fallback);
    }
}

/// Builds the complete [`ManifoldGeometry`] for `cfg` and `sandbox`, respecting `MAX_MANIFOLD_VERTICES`.
#[must_use]
pub fn build_manifold_geometry(
    cfg: &RiemannianStudioConfig,
    sandbox: &CustomSurfaceSandbox,
) -> ManifoldGeometry {
    let mut nu = cfg.u_res.clamp(8, 62);
    let mut nv = cfg.v_res.clamp(8, 62);
    while (nu + 1) * (nv + 1) > MAX_MANIFOLD_VERTICES && nv > 8 {
        nv -= 1;
        if (nu + 1) * (nv + 1) > MAX_MANIFOLD_VERTICES && nu > 8 {
            nu -= 1;
        }
    }
    match cfg.manifold {
        ManifoldKind::Sphere => build_polar_manifold(nu, nv, false, cfg.sulcal_depth),
        ManifoldKind::FoldedCorticalShell => build_polar_manifold(nu, nv, true, cfg.sulcal_depth),
        ManifoldKind::Torus => {
            let torus = Torus {
                major_radius: 1.65,
                minor_radius: 0.62,
            };
            build_periodic_manifold(&torus, nu, nv)
        }
        ManifoldKind::KleinBottle => {
            let klein = KleinBottle { radius: 1.35 };
            build_open_or_klein_manifold(&klein, ((0.0, TAU), (0.0, TAU)), nu, nv, true)
        }
        ManifoldKind::CustomHeightfield => {
            let surf = sandbox.build_heightfield();
            let domain = sandbox.uv_domain_for(ParametricSurfaceKind::CustomHeightfield);
            build_open_or_klein_manifold(&surf, domain, nu, nv, false)
        }
        ManifoldKind::CustomParametric => {
            let surf = sandbox.build_parametric();
            let domain = sandbox.uv_domain_for(ParametricSurfaceKind::CustomParametric);
            build_open_or_klein_manifold(&surf, domain, nu, nv, false)
        }
    }
}

fn build_polar_manifold(
    nu: usize,
    nv: usize,
    folded_cortex: bool,
    sulcal_depth: f64,
) -> ManifoldGeometry {
    let sphere = Sphere { radius: 1.25 };
    let sample = |u: f64, v: f64| -> ([f64; 3], f64) {
        if folded_cortex {
            let (w, _p, _t) = cortical_shell_point_and_thickness(u, v, sulcal_depth);
            let k_est = 0.65 + 1.8 * sulcal_depth * (5.0 * u).sin() * (4.0 * v).sin();
            (w, k_est)
        } else {
            let p = sphere.position(u, v);
            (
                [p.x, p.y, p.z],
                sphere.gaussian_curvature(u, v.clamp(1e-3, PI - 1e-3)),
            )
        }
    };

    let num_verts = 2 + (nv - 1) * nu;
    let mut positions = Vec::with_capacity(num_verts);
    let mut uvs = Vec::with_capacity(num_verts);
    let mut vertex_grid_ij = Vec::with_capacity(num_verts);
    let mut gaussian_curvature = Vec::with_capacity(num_verts);

    let (north, k_n) = sample(0.0, 0.0);
    let (south, k_s) = sample(0.0, PI);
    positions.push(north);
    uvs.push((0.0, 0.0));
    vertex_grid_ij.push((0, 0));
    gaussian_curvature.push(k_n);

    positions.push(south);
    uvs.push((0.0, PI));
    vertex_grid_ij.push((0, nv));
    gaussian_curvature.push(k_s);

    let mut grid_pts = vec![vec![Point3::origin(); nv + 1]; nu];
    for row in grid_pts.iter_mut().take(nu) {
        row[0] = Point3::new(north[0], north[1], north[2]);
        row[nv] = Point3::new(south[0], south[1], south[2]);
    }

    for j in 1..nv {
        let v = PI * (j as f64 / nv as f64);
        for (i, row) in grid_pts.iter_mut().enumerate().take(nu) {
            let u = TAU * (i as f64 / nu as f64);
            let (p, k_val) = sample(u, v);
            row[j] = Point3::new(p[0], p[1], p[2]);
            positions.push(p);
            uvs.push((u, v));
            vertex_grid_ij.push((i, j));
            gaussian_curvature.push(if k_val.is_finite() { k_val } else { 0.0 });
        }
    }

    let triangles = build_polar_triangles(nu, nv);

    finalize_geometry(RawManifoldPatch {
        positions,
        triangles,
        uvs,
        vertex_grid_ij,
        grid_dims: (nu, nv + 1),
        uv_domain: ((0.0, TAU), (0.0, PI)),
        two_sided: false,
        gaussian_curvature,
        discrete_surface: DiscreteSurface::new(grid_pts, true, false),
    })
}

fn build_polar_triangles(nu: usize, nv: usize) -> Vec<[usize; 3]> {
    let ring = |j: usize, i: usize| -> usize { 2 + (j - 1) * nu + (i % nu) };
    let mut triangles = Vec::with_capacity(2 * (nv - 1) * nu);
    for i in 0..nu {
        triangles.push([0, ring(1, i), ring(1, i + 1)]);
    }
    for j in 1..(nv - 1) {
        for i in 0..nu {
            let (r00, r10, r11, r01) = (
                ring(j, i),
                ring(j + 1, i),
                ring(j + 1, i + 1),
                ring(j, i + 1),
            );
            triangles.push([r00, r10, r11]);
            triangles.push([r00, r11, r01]);
        }
    }
    for i in 0..nu {
        triangles.push([1, ring(nv - 1, i + 1), ring(nv - 1, i)]);
    }
    triangles
}

fn build_periodic_manifold<S: SurfaceAnalysis + ParametricSurface>(
    surface: &S,
    nu: usize,
    nv: usize,
) -> ManifoldGeometry {
    let mut positions = Vec::with_capacity(nu * nv);
    let mut uvs = Vec::with_capacity(nu * nv);
    let mut vertex_grid_ij = Vec::with_capacity(nu * nv);
    let mut gaussian_curvature = Vec::with_capacity(nu * nv);
    let mut grid_pts = vec![vec![Point3::origin(); nv]; nu];

    for (i, row) in grid_pts.iter_mut().enumerate().take(nu) {
        let u = TAU * (i as f64 / nu as f64);
        for (j, slot) in row.iter_mut().enumerate().take(nv) {
            let v = TAU * (j as f64 / nv as f64);
            let p = surface.position(u, v);
            let pos = if p.coords.iter().all(|c| c.is_finite()) {
                [p.x, p.y, p.z]
            } else {
                [0.0, 0.0, 0.0]
            };
            *slot = Point3::new(pos[0], pos[1], pos[2]);
            positions.push(pos);
            uvs.push((u, v));
            vertex_grid_ij.push((i, j));
            let k = surface.gaussian_curvature(u, v);
            gaussian_curvature.push(if k.is_finite() {
                k.clamp(-50.0, 50.0)
            } else {
                0.0
            });
        }
    }

    let idx = |i: usize, j: usize| (i % nu) * nv + (j % nv);
    let mut triangles = Vec::with_capacity(2 * nu * nv);
    for i in 0..nu {
        for j in 0..nv {
            let (v00, v10, v11, v01) = (idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1));
            triangles.push([v00, v10, v11]);
            triangles.push([v00, v11, v01]);
        }
    }

    finalize_geometry(RawManifoldPatch {
        positions,
        triangles,
        uvs,
        vertex_grid_ij,
        grid_dims: (nu, nv),
        uv_domain: ((0.0, TAU), (0.0, TAU)),
        two_sided: false,
        gaussian_curvature,
        discrete_surface: DiscreteSurface::new(grid_pts, true, true),
    })
}

fn build_open_or_klein_manifold<S: SurfaceAnalysis + ParametricSurface>(
    surface: &S,
    domain: ((f64, f64), (f64, f64)),
    nu: usize,
    nv: usize,
    wrap_v: bool,
) -> ManifoldGeometry {
    let ((u0, u1), (v0, v1)) = domain;
    let (rows, cols) = (nu + 1, if wrap_v { nv } else { nv + 1 });
    let mut positions = Vec::with_capacity(rows * cols);
    let mut uvs = Vec::with_capacity(rows * cols);
    let mut vertex_grid_ij = Vec::with_capacity(rows * cols);
    let mut gaussian_curvature = Vec::with_capacity(rows * cols);
    let mut grid_pts = vec![vec![Point3::origin(); cols]; rows];

    for (i, row) in grid_pts.iter_mut().enumerate().take(rows) {
        let u = u0 + (u1 - u0) * (i as f64 / nu as f64);
        for (j, slot) in row.iter_mut().enumerate().take(cols) {
            let v = v0 + (v1 - v0) * (j as f64 / nv as f64);
            let p = surface.position(u, v);
            let pos = if p.coords.iter().all(|c| c.is_finite()) {
                [
                    p.x.clamp(-20.0, 20.0),
                    p.y.clamp(-20.0, 20.0),
                    p.z.clamp(-20.0, 20.0),
                ]
            } else {
                [0.0, 0.0, 0.0]
            };
            *slot = Point3::new(pos[0], pos[1], pos[2]);
            positions.push(pos);
            uvs.push((u, v));
            vertex_grid_ij.push((i, j));
            let k = surface.gaussian_curvature(u, v);
            gaussian_curvature.push(if k.is_finite() {
                k.clamp(-50.0, 50.0)
            } else {
                0.0
            });
        }
    }

    let at = |i: usize, j: usize| i * cols + (if wrap_v { j % cols } else { j });
    let mut triangles = Vec::with_capacity(2 * nu * nv);
    let j_limit = if wrap_v { cols } else { nv };
    for i in 0..nu {
        for j in 0..j_limit {
            let (v00, v10, v11, v01) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
            triangles.push([v00, v10, v11]);
            triangles.push([v00, v11, v01]);
        }
    }

    finalize_geometry(RawManifoldPatch {
        positions,
        triangles,
        uvs,
        vertex_grid_ij,
        grid_dims: (rows, cols),
        uv_domain: domain,
        two_sided: true,
        gaussian_curvature,
        discrete_surface: DiscreteSurface::new(grid_pts, false, wrap_v),
    })
}

struct RawManifoldPatch {
    positions: Vec<[f64; 3]>,
    triangles: Vec<[usize; 3]>,
    uvs: Vec<(f64, f64)>,
    vertex_grid_ij: Vec<(usize, usize)>,
    grid_dims: (usize, usize),
    uv_domain: ((f64, f64), (f64, f64)),
    two_sided: bool,
    gaussian_curvature: Vec<f64>,
    discrete_surface: DiscreteSurface,
}

fn finalize_geometry(patch: RawManifoldPatch) -> ManifoldGeometry {
    let n = patch.positions.len();
    let mut normals = vec![[0.0, 0.0, 1.0]; n];
    recompute_vertex_normals(&patch.positions, &patch.triangles, &mut normals);
    let laplacian = CotangentLaplacian::from_indexed(&patch.positions, &patch.triangles);
    let mut mean_curvature = vec![0.0; n];
    laplacian.compute_mean_curvature(&patch.positions, &mut mean_curvature);
    ManifoldGeometry {
        positions: patch.positions,
        normals,
        triangles: patch.triangles,
        uvs: patch.uvs,
        vertex_grid_ij: patch.vertex_grid_ij,
        grid_dims: patch.grid_dims,
        uv_domain: patch.uv_domain,
        two_sided: patch.two_sided,
        mean_curvature,
        gaussian_curvature: patch.gaussian_curvature,
        laplacian,
        discrete_surface: patch.discrete_surface,
    }
}
