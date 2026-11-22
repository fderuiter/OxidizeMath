#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Cold-path toolpath generation, millimeter bed projection, and travel-distance metrics
//! for [`super::FabricationStudioTool`].

use super::state::{
    FabricationSourceKind, FabricationStudioState, StudioFlowElement, StudioFlowElementKind,
};
use egui_plot::{FabricationBedTransform, FabricationPolyline};
use math_commons::contours::{
    extract_contours, extract_multi_contours, order_paths_nearest_neighbor, pen_lift_segments,
    pen_lift_travel_distance, ContourPath, GridBounds2D, Segment2D,
};
use math_explorer::pure_math::tilings::generate_aperiodic_tiling;
use std::f64::consts::TAU;

/// Projected physical millimeter toolpaths and pen-travel telemetry for the active studio config.
#[derive(Clone, Debug)]
pub struct FabricationArtifact {
    /// Physical bed width in millimeters.
    pub bed_width_mm: f64,
    /// Physical bed height in millimeters.
    pub bed_height_mm: f64,
    /// Safe inset margin in millimeters.
    pub margin_mm: f64,
    /// `#layer-cut` paths projected into bed millimeters.
    pub cut_paths_mm: Vec<FabricationPolyline>,
    /// `#layer-score` paths projected into bed millimeters.
    pub score_paths_mm: Vec<FabricationPolyline>,
    /// Rapid non-cutting pen-lift jump segments in millimeters.
    pub pen_lifts_mm: Vec<Segment2D>,
    /// Total cut path arc length in millimeters.
    pub cut_length_mm: f64,
    /// Total score path arc length in millimeters.
    pub score_length_mm: f64,
    /// Unoptimized rapid pen-lift travel distance in millimeters.
    pub raw_travel_mm: f64,
    /// Active (nearest-neighbor optimized when enabled) pen-lift travel distance in millimeters.
    pub active_travel_mm: f64,
}

/// Builds the complete millimeter [`FabricationArtifact`] for `state`.
pub fn build_fabrication_artifact(state: &FabricationStudioState) -> Result<FabricationArtifact, String> {
    let (domain_bounds, raw_cut, raw_score) = match state.source {
        FabricationSourceKind::TuringField => generate_turing_domain_paths(state)?,
        FabricationSourceKind::PotentialFlow => generate_flow_domain_paths(state)?,
        FabricationSourceKind::AperiodicTiling => generate_tiling_domain_paths(state)?,
    };

    let options = state.svg_options();
    let transform = FabricationBedTransform::from_domain_bounds(domain_bounds, &options)?;
    let unopt_score_mm = project_contour_paths(&transform, &raw_score);
    let unopt_cut_mm = project_contour_paths(&transform, &raw_cut);

    let mut raw_sequence = unopt_score_mm.clone();
    raw_sequence.extend(unopt_cut_mm.iter().cloned());
    let raw_travel_mm = pen_lift_travel_distance(&raw_sequence);

    let score_contours_mm = if state.optimize_travel {
        order_paths_nearest_neighbor(&unopt_score_mm)
    } else {
        unopt_score_mm
    };
    let cut_contours_mm = if state.optimize_travel {
        order_paths_nearest_neighbor(&unopt_cut_mm)
    } else {
        unopt_cut_mm
    };

    let mut active_sequence = score_contours_mm.clone();
    active_sequence.extend(cut_contours_mm.iter().cloned());
    let active_travel_mm = pen_lift_travel_distance(&active_sequence);
    let pen_lifts_mm = pen_lift_segments(&active_sequence);

    let cut_length_mm: f64 = cut_contours_mm.iter().map(ContourPath::arc_length).sum();
    let score_length_mm: f64 = score_contours_mm.iter().map(ContourPath::arc_length).sum();

    Ok(FabricationArtifact {
        bed_width_mm: transform.width_mm,
        bed_height_mm: transform.height_mm,
        margin_mm: transform.margin_mm,
        cut_paths_mm: to_fabrication_polylines(&cut_contours_mm),
        score_paths_mm: to_fabrication_polylines(&score_contours_mm),
        pen_lifts_mm,
        cut_length_mm,
        score_length_mm,
        raw_travel_mm,
        active_travel_mm,
    })
}

fn project_contour_paths(
    transform: &FabricationBedTransform,
    paths: &[ContourPath],
) -> Vec<ContourPath> {
    paths
        .iter()
        .filter_map(|p| {
            let pts: Vec<[f64; 2]> = p
                .points
                .iter()
                .copied()
                .filter(|pt| pt[0].is_finite() && pt[1].is_finite())
                .map(|pt| transform.project_point(pt))
                .collect();
            (pts.len() >= 2).then_some(ContourPath::new(pts, p.closed))
        })
        .collect()
}

fn to_fabrication_polylines(paths: &[ContourPath]) -> Vec<FabricationPolyline> {
    paths
        .iter()
        .map(|p| FabricationPolyline::new(p.points.clone(), p.closed))
        .collect()
}

type DomainPathsResult = Result<([f64; 4], Vec<ContourPath>, Vec<ContourPath>), String>;

fn generate_tiling_domain_paths(state: &FabricationStudioState) -> DomainPathsResult {
    let patch = generate_aperiodic_tiling(state.tiling_kind.to_domain_kind(), state.tiling_depth)
        .map_err(|e| e.to_string())?;
    let bounds = [
        patch.bounds.min_x,
        patch.bounds.max_x,
        patch.bounds.min_y,
        patch.bounds.max_y,
    ];
    Ok((bounds, patch.boundary_paths, patch.interior_paths))
}

fn generate_turing_domain_paths(state: &FabricationStudioState) -> DomainPathsResult {
    let n = state.turing_grid_size;
    let u_grid = simulate_deterministic_turing_grid(state);
    let bounds = GridBounds2D::new(0.0, (n - 1) as f64, 0.0, (n - 1) as f64)
        .map_err(|e| e.to_string())?;
    let (min_u, max_u) = u_grid
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let span = (max_u - min_u).max(1e-6);

    let mut score_levels = Vec::with_capacity(state.turing_score_levels);
    for idx in 0..state.turing_score_levels {
        let q = 0.22 + 0.56 * ((idx as f64) + 0.5) / (state.turing_score_levels as f64);
        score_levels.push(min_u + q * span);
    }
    let multi = extract_multi_contours(&u_grid, n, n, &bounds, &score_levels)
        .map_err(|e| e.to_string())?;
    let mut score_paths: Vec<ContourPath> = multi.into_iter().flat_map(|(_, p)| p).collect();

    let mut cut_paths = Vec::new();
    if state.turing_coaster_border {
        let center = 0.5 * ((n - 1) as f64);
        let radius = center * 1.02;
        cut_paths.push(circular_coaster_loop([center, center], radius, 96));
        clip_paths_inside_circle(&mut score_paths, [center, center], radius * 0.96);
    } else {
        let mid_iso = min_u + 0.5 * span;
        cut_paths = extract_contours(&u_grid, n, n, &bounds, mid_iso).map_err(|e| e.to_string())?;
    }

    let dom = if state.turing_coaster_border {
        let pad = 0.04 * ((n - 1) as f64);
        [-pad, ((n - 1) as f64) + pad, -pad, ((n - 1) as f64) + pad]
    } else {
        [bounds.min_x, bounds.max_x, bounds.min_y, bounds.max_y]
    };
    Ok((dom, cut_paths, score_paths))
}

fn circular_coaster_loop(center: [f64; 2], radius: f64, segments: usize) -> ContourPath {
    let mut pts = Vec::with_capacity(segments);
    for idx in 0..segments {
        let ang = TAU * (idx as f64) / (segments as f64);
        pts.push([center[0] + radius * ang.cos(), center[1] + radius * ang.sin()]);
    }
    ContourPath::new(pts, true)
}

fn clip_paths_inside_circle(paths: &mut Vec<ContourPath>, center: [f64; 2], max_r: f64) {
    let r2 = max_r * max_r;
    let orig = std::mem::take(paths);
    for path in orig {
        let all_inside = path.points.iter().all(|p| {
            let dx = p[0] - center[0];
            let dy = p[1] - center[1];
            dx * dx + dy * dy <= r2
        });
        if all_inside {
            paths.push(path);
            continue;
        }
        let mut run = Vec::new();
        for pt in path.polyline_points() {
            let dx = pt[0] - center[0];
            let dy = pt[1] - center[1];
            if dx * dx + dy * dy <= r2 {
                run.push(pt);
            } else if run.len() >= 2 {
                paths.push(ContourPath::new(std::mem::take(&mut run), false));
            } else {
                run.clear();
            }
        }
        if run.len() >= 2 {
            paths.push(ContourPath::new(run, false));
        }
    }
}

fn cfl_substep_plan(state: &FabricationStudioState) -> (usize, f64) {
    let d_max = state.turing_d_u.max(state.turing_d_v).max(1.0);
    let safe_dt = 0.20 / d_max;
    let substeps = ((state.turing_dt / safe_dt).ceil() as usize).clamp(1, 64);
    (substeps, state.turing_dt / (substeps as f64))
}

#[cfg(feature = "biology")]
fn simulate_deterministic_turing_grid(state: &FabricationStudioState) -> Vec<f64> {
    use math_explorer::biology::diffusion::FiniteDifference2D;
    use math_explorer::biology::morphogenesis::{DiffusionCoeff, SchnakenbergKinetics, TuringSystem};
    use math_explorer::math_kernel::types::{Dimension, StepSize};

    let n = state.turing_grid_size;
    let kinetics = SchnakenbergKinetics {
        a: state.turing_a,
        b: state.turing_b,
    };
    let diffusion =
        FiniteDifference2D::new(Dimension(n), Dimension(n), StepSize(1.0), StepSize(1.0));
    let mut system = TuringSystem::new_with_kinetics(
        Dimension(n * n),
        DiffusionCoeff(state.turing_d_u),
        DiffusionCoeff(state.turing_d_v),
        kinetics,
        diffusion,
    );
    let mut u_init = vec![0.0; n * n];
    let mut v_init = vec![0.0; n * n];
    seed_schnakenberg_grid(&mut u_init, &mut v_init, n, state.turing_a, state.turing_b);
    system.u_mut().copy_from_slice(&u_init);
    system.v_mut().copy_from_slice(&v_init);
    let (substeps, sub_dt) = cfl_substep_plan(state);
    for _ in 0..(state.turing_steps * substeps) {
        system.step(sub_dt);
    }
    system.u().to_vec()
}

#[cfg(not(feature = "biology"))]
fn simulate_deterministic_turing_grid(state: &FabricationStudioState) -> Vec<f64> {
    let n = state.turing_grid_size;
    let mut u = vec![0.0; n * n];
    let mut v = vec![0.0; n * n];
    seed_schnakenberg_grid(&mut u, &mut v, n, state.turing_a, state.turing_b);
    let mut next_u = vec![0.0; n * n];
    let mut next_v = vec![0.0; n * n];
    let (substeps, sub_dt) = cfl_substep_plan(state);
    for _ in 0..(state.turing_steps * substeps) {
        step_schnakenberg_fallback(&u, &v, &mut next_u, &mut next_v, n, state, sub_dt);
        std::mem::swap(&mut u, &mut next_u);
        std::mem::swap(&mut v, &mut next_v);
    }
    u
}

#[cfg(not(feature = "biology"))]
fn step_schnakenberg_fallback(
    u: &[f64],
    v: &[f64],
    next_u: &mut [f64],
    next_v: &mut [f64],
    n: usize,
    s: &FabricationStudioState,
    dt: f64,
) {
    use math_commons::math_kernel::types::flatten_2d_index;

    for iy in 0..n {
        let ym = if iy == 0 { n - 1 } else { iy - 1 };
        let yp = (iy + 1) % n;
        for ix in 0..n {
            let xm = if ix == 0 { n - 1 } else { ix - 1 };
            let xp = (ix + 1) % n;
            let idx = flatten_2d_index(ix, iy, n);
            let left = flatten_2d_index(xm, iy, n);
            let right = flatten_2d_index(xp, iy, n);
            let down = flatten_2d_index(ix, ym, n);
            let up = flatten_2d_index(ix, yp, n);
            let uc = u[idx];
            let vc = v[idx];
            let lap_u = u[left] + u[right] + u[down] + u[up] - 4.0 * uc;
            let lap_v = v[left] + v[right] + v[down] + v[up] - 4.0 * vc;
            let u2v = uc * uc * vc;
            next_u[idx] =
                (uc + dt * (s.turing_d_u * lap_u + s.turing_a - uc + u2v)).clamp(0.0, 20.0);
            next_v[idx] = (vc + dt * (s.turing_d_v * lap_v + s.turing_b - u2v)).clamp(0.0, 20.0);
        }
    }
}

fn seed_schnakenberg_grid(u: &mut [f64], v: &mut [f64], n: usize, a: f64, b: f64) {
    let u0 = (a + b).max(0.05);
    let v0 = b / (u0 * u0);
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    for idx in 0..(n * n) {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let noise = (((seed >> 33) as f64) / (u32::MAX as f64)) - 0.5;
        u[idx] = (u0 + 0.08 * noise).max(1e-3);
        v[idx] = (v0 - 0.08 * noise).max(1e-3);
    }
}

fn generate_flow_domain_paths(state: &FabricationStudioState) -> DomainPathsResult {
    let vr = state.flow_view_range;
    let traced = trace_studio_flow_paths(&state.flow_elements, vr, state.flow_streamline_count);
    let mut cut_paths = Vec::new();
    let mut score_paths = Vec::new();
    for path in traced {
        if path.closed {
            cut_paths.push(path);
        } else {
            score_paths.push(path);
        }
    }
    cut_paths.push(ContourPath::new(
        vec![[-vr, -vr], [vr, -vr], [vr, vr], [-vr, vr]],
        true,
    ));
    Ok(([-vr, vr, -vr, vr], cut_paths, score_paths))
}

#[cfg(feature = "physics")]
fn trace_studio_flow_paths(
    elements: &[StudioFlowElement],
    view_range: f64,
    count: usize,
) -> Vec<ContourPath> {
    use math_explorer::physics::fluid_dynamics::potential_flow::{
        Doublet, FlowElement, PotentialFlowField, Source, UniformFlow, Vortex,
    };

    let mut field = PotentialFlowField::new();
    for el in elements {
        let boxed: Box<dyn FlowElement> = match el.kind {
            StudioFlowElementKind::Uniform => Box::new(UniformFlow::new(el.strength, el.angle_deg)),
            StudioFlowElementKind::Source => Box::new(Source::new(el.strength, el.x, el.y)),
            StudioFlowElementKind::Sink => Box::new(Source::new(-el.strength, el.x, el.y)),
            StudioFlowElementKind::Vortex => Box::new(Vortex::new(el.strength, el.x, el.y)),
            StudioFlowElementKind::Doublet => Box::new(Doublet::new(el.strength, el.x, el.y)),
        };
        field.add(boxed);
    }
    field.trace_streamlines(view_range, count)
}

#[cfg(not(feature = "physics"))]
fn trace_studio_flow_paths(
    elements: &[StudioFlowElement],
    view_range: f64,
    count: usize,
) -> Vec<ContourPath> {
    use math_commons::contours::{trace_streamlines_rk4, StreamlineConfig};
    use std::f64::consts::PI;

    let Ok(bounds) = GridBounds2D::new(-view_range, view_range, -view_range, view_range) else {
        return Vec::new();
    };
    let Ok(cfg) = StreamlineConfig::new(bounds, (view_range * 0.035).max(1e-3), 450, 1e-4) else {
        return Vec::new();
    };
    let seeds: Vec<[f64; 2]> = (0..count)
        .map(|i| {
            let frac = ((i as f64) + 0.5) / (count as f64);
            [-view_range * 0.96, -view_range + 2.0 * view_range * frac]
        })
        .collect();
    trace_streamlines_rk4(&seeds, &cfg, |x, y| {
        let mut vx = 0.0;
        let mut vy = 0.0;
        for el in elements {
            let (dx, dy) = (x - el.x, y - el.y);
            let r2 = dx * dx + dy * dy;
            match el.kind {
                StudioFlowElementKind::Uniform => {
                    let rad = el.angle_deg.to_radians();
                    vx += el.strength * rad.cos();
                    vy += el.strength * rad.sin();
                }
                StudioFlowElementKind::Source | StudioFlowElementKind::Sink if r2 > 1e-8 => {
                    let s = if el.kind == StudioFlowElementKind::Sink {
                        -el.strength
                    } else {
                        el.strength
                    };
                    vx += s * dx / (2.0 * PI * r2);
                    vy += s * dy / (2.0 * PI * r2);
                }
                StudioFlowElementKind::Vortex if r2 > 1e-8 => {
                    vx -= el.strength * dy / (2.0 * PI * r2);
                    vy += el.strength * dx / (2.0 * PI * r2);
                }
                StudioFlowElementKind::Doublet if r2 > 1e-8 => {
                    let r4 = r2 * r2;
                    vx -= el.strength * (dx * dx - dy * dy) / (2.0 * PI * r4);
                    vy -= el.strength * (2.0 * dx * dy) / (2.0 * PI * r4);
                }
                _ => {}
            }
        }
        [vx, vy]
    })
    .unwrap_or_default()
}
