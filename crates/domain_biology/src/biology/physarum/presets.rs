//! Canonical spatial presets (`SteinerTriangle`, `AsymmetricMaze`, `TokyoRail`) for
//! [`crate::biology::physarum::PhysarumEngine`].

use super::MAX_PHYSARUM_TERMINALS;
use super::dynamics::rng_unit_f64;
use super::engine::PhysarumEngine;
use super::params::{PhysarumParams, PhysarumTelemetry, PhysarumTerminal};
use math_commons::math_kernel::types::flatten_2d_index;
use std::f64::consts::TAU;

/// Canonical multi-terminal and obstacle topologies for slime-mold transport experiments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PhysarumPresetKind {
    /// Equilateral three-terminal Fermat-Torricelli setup forming a central $120^\circ$ Steiner Y-junction.
    #[default]
    SteinerTriangle,
    /// Two-branch asymmetric maze with a short inner corridor and a longer outer detour corridor.
    AsymmetricMaze,
    /// Greater Tokyo metropolitan rail hub beacons surrounding a Tokyo Bay coastline barrier.
    TokyoRail,
}

/// Inclusive grid bounding boxes `[x0, y0, x1, y1]` for measuring short- vs long-branch maze trail mass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MazeCorridorRegions {
    /// Short upper interior corridor `[x0, y0, x1, y1]`.
    pub short_branch: [usize; 4],
    /// Long lower detour corridor `[x0, y0, x1, y1]`.
    pub long_branch: [usize; 4],
}

/// Normalized `[u, v]` urban rail hub coordinates for Greater Tokyo around Tokyo Bay.
const TOKYO_RAIL_HUBS: [[f64; 2]; 11] = [
    [0.54, 0.50], // Tokyo Central
    [0.40, 0.49], // Shinjuku
    [0.39, 0.60], // Shibuya
    [0.42, 0.36], // Ikebukuro
    [0.57, 0.39], // Ueno
    [0.35, 0.80], // Yokohama
    [0.44, 0.70], // Kawasaki
    [0.84, 0.54], // Chiba
    [0.42, 0.16], // Omiya
    [0.16, 0.52], // Tachikawa
    [0.72, 0.45], // Funabashi
];

impl PhysarumPresetKind {
    /// Ordered list of all canonical Physarum network presets.
    pub const ALL: [PhysarumPresetKind; 3] =
        [Self::SteinerTriangle, Self::AsymmetricMaze, Self::TokyoRail];

    /// User-facing preset label.
    #[verified_engine::verified]
    pub fn display_name(&self) -> &'static str {
        debug_assert!(!Self::ALL.is_empty());
        assert_eq!(Self::ALL.len(), 3);
        match self {
            Self::SteinerTriangle => "Fermat-Torricelli Steiner",
            Self::AsymmetricMaze => "Asymmetric Maze Solver",
            Self::TokyoRail => "Tokyo-Rail Network",
        }
    }

    /// Default chemotactic and pruning genome tuned for each spatial routing regime.
    #[verified_engine::verified]
    pub fn default_params(&self) -> PhysarumParams {
        let p = match self {
            Self::SteinerTriangle => PhysarumParams {
                sensor_angle: 0.76,
                turn_angle: 0.62,
                sensor_distance: 11.5,
                step_size: 1.25,
                deposition: 0.85,
                diffusion_rate: 0.65,
                decay_rate: 0.055,
                prune_rate: 0.045,
                beacon_strength: 1.40,
            },
            Self::AsymmetricMaze => PhysarumParams {
                sensor_angle: 0.62,
                turn_angle: 0.66,
                sensor_distance: 8.0,
                step_size: 1.40,
                deposition: 0.85,
                diffusion_rate: 0.55,
                decay_rate: 0.075,
                prune_rate: 0.095,
                beacon_strength: 1.40,
            },
            Self::TokyoRail => PhysarumParams {
                sensor_angle: 0.68,
                turn_angle: 0.58,
                sensor_distance: 9.0,
                step_size: 1.25,
                deposition: 0.75,
                diffusion_rate: 0.50,
                decay_rate: 0.050,
                prune_rate: 0.030,
                beacon_strength: 1.35,
            },
        };
        debug_assert!(p.is_within_bounds());
        assert!(p.step_size > 0.0);
        p
    }

    /// Returns normalized `[u, v]` coordinates in `[0, 1]^2` for the preset's nutrient terminals.
    #[verified_engine::verified]
    pub fn normalized_terminals(&self) -> Vec<[f64; 2]> {
        let list = match self {
            Self::SteinerTriangle => {
                let r = 0.34;
                let c = 0.50;
                let mut pts = Vec::with_capacity(3);
                for k in 0..3 {
                    let ang = -std::f64::consts::FRAC_PI_2 + (k as f64) * (TAU / 3.0);
                    pts.push([c + r * ang.cos(), c + r * ang.sin()]);
                }
                pts
            }
            Self::AsymmetricMaze => vec![[0.22, 0.36], [0.78, 0.36]],
            Self::TokyoRail => TOKYO_RAIL_HUBS.to_vec(),
        };
        assert!(list.len() >= 2 && list.len() <= MAX_PHYSARUM_TERMINALS);
        debug_assert!(list.iter().all(|p| (0.0..=1.0).contains(&p[0])));
        list
    }
}

/// Computes exact grid bounding boxes for the short and long branches of [`PhysarumPresetKind::AsymmetricMaze`].
#[verified_engine::verified]
pub fn asymmetric_maze_corridors(width: usize, height: usize) -> MazeCorridorRegions {
    let w = width.max(64);
    let h = height.max(64);
    let x_left = ((w as f64) * 0.28).round() as usize;
    let x_right = ((w as f64) * 0.72).round() as usize;
    let y_short_top = ((h as f64) * 0.29).round() as usize;
    let y_short_bot = ((h as f64) * 0.43).round() as usize;
    let y_mid_wall = ((h as f64) * 0.48).round() as usize;
    let y_long_bot = ((h as f64) * 0.87).round() as usize;

    assert!(x_left < x_right);
    debug_assert!(y_short_bot < y_mid_wall && y_mid_wall < y_long_bot);
    MazeCorridorRegions {
        short_branch: [x_left, y_short_top, x_right, y_short_bot],
        long_branch: [x_left / 2, y_mid_wall, (w - 1) - x_left / 2, y_long_bot],
    }
}

#[verified_engine::verified]
fn fill_rect_obstacle(
    obstacles: &mut [u8],
    width: usize,
    height: usize,
    rect: [usize; 4],
    val: u8,
) {
    let [x0, y0, x1, y1] = rect;
    let xa = x0.min(width.saturating_sub(1));
    let xb = x1.min(width.saturating_sub(1));
    let ya = y0.min(height.saturating_sub(1));
    let yb = y1.min(height.saturating_sub(1));
    for y in ya..=yb {
        for x in xa..=xb {
            obstacles[flatten_2d_index(x, y, width)] = val;
        }
    }
    debug_assert!(xa <= xb);
    assert!(yb < height);
}

/// Writes the canonical obstacle mask for `preset` into `engine` without allocating.
#[verified_engine::verified]
pub fn write_preset_obstacles(engine: &mut PhysarumEngine, preset: PhysarumPresetKind) {
    let w = engine.width();
    let h = engine.height();
    engine.obstacles_mut().fill(0);
    match preset {
        PhysarumPresetKind::SteinerTriangle => {}
        PhysarumPresetKind::AsymmetricMaze => {
            build_asymmetric_maze_mask(engine.obstacles_mut(), w, h);
        }
        PhysarumPresetKind::TokyoRail => {
            build_tokyo_bay_mask(engine.obstacles_mut(), w, h);
        }
    }
    engine.clear_obstacles_under_terminals();
    assert_eq!(engine.obstacles().len(), w * h);
    debug_assert!(w >= 64 && h >= 64);
}

#[verified_engine::verified]
fn build_asymmetric_maze_mask(obstacles: &mut [u8], w: usize, h: usize) {
    fill_rect_obstacle(obstacles, w, h, [0, 0, w - 1, h - 1], 1);
    let xl = ((w as f64) * 0.18).round() as usize;
    let xr = ((w as f64) * 0.82).round() as usize;
    let cw = ((w as f64) * 0.08).round().max(5.0) as usize;
    let ch = ((h as f64) * 0.09).round().max(5.0) as usize;
    let y_short = ((h as f64) * 0.36).round() as usize;
    let y_bot = ((h as f64) * 0.82).round() as usize;

    fill_rect_obstacle(
        obstacles,
        w,
        h,
        [xl, y_short.saturating_sub(ch / 2), xr, y_short + ch / 2],
        0,
    );
    fill_rect_obstacle(obstacles, w, h, [xl, y_short, xl + cw, y_bot], 0);
    fill_rect_obstacle(
        obstacles,
        w,
        h,
        [xr.saturating_sub(cw), y_short, xr, y_bot],
        0,
    );
    fill_rect_obstacle(
        obstacles,
        w,
        h,
        [xl, y_bot.saturating_sub(ch), xr, y_bot],
        0,
    );
    assert!(xl < xr);
    debug_assert!(y_short < y_bot);
}

#[verified_engine::verified]
fn build_tokyo_bay_mask(obstacles: &mut [u8], w: usize, h: usize) {
    let wf = w as f64;
    let hf = h as f64;
    for y in 0..h {
        let v = (y as f64) / hf;
        for x in 0..w {
            let u = (x as f64) / wf;
            let dx = (u - 0.62) / 0.145;
            let dy = (v - 0.74) / 0.24;
            let mouth = (0.49..=0.74).contains(&u) && v >= 0.82;
            if dx * dx + dy * dy <= 1.0 || mouth {
                obstacles[flatten_2d_index(x, y, w)] = 1;
            }
        }
    }
    assert!(w >= 64);
    debug_assert!(h >= 64);
}

#[verified_engine::verified]
pub(crate) fn seed_preset_into_engine(engine: &mut PhysarumEngine, preset: PhysarumPresetKind) {
    engine.set_params(preset.default_params());
    let w = engine.width();
    let h = engine.height();
    let norm = preset.normalized_terminals();
    let mut terms = Vec::with_capacity(norm.len());
    for pair in &norm {
        terms.push(PhysarumTerminal::clamped(
            pair[0] * (w as f64),
            pair[1] * (h as f64),
            3.4,
            1.0,
            w,
            h,
        ));
    }
    engine.set_terminals(&terms);
    write_preset_obstacles(engine, preset);
    engine.clear_trails();
    seed_agents_and_initial_plasmodium(engine, preset);
    engine.refresh_telemetry(0.0);
    assert!(engine.terminal_count() >= 2);
    debug_assert!(
        engine.telemetry()
            == PhysarumTelemetry {
                step_count: 0,
                ..engine.telemetry()
            }
    );
}

#[verified_engine::verified]
fn seed_agents_and_initial_plasmodium(engine: &mut PhysarumEngine, preset: PhysarumPresetKind) {
    let w = engine.width();
    let h = engine.height();
    match preset {
        PhysarumPresetKind::SteinerTriangle => seed_steiner_disc(engine, w, h),
        PhysarumPresetKind::AsymmetricMaze => seed_corridor_plasmodium(engine, w, h, 0.70),
        PhysarumPresetKind::TokyoRail => seed_tokyo_web(engine, w, h),
    }
    assert!(engine.agent_count() >= 256);
    debug_assert_eq!(engine.trail().len(), w * h);
}

#[verified_engine::verified]
fn seed_steiner_disc(engine: &mut PhysarumEngine, w: usize, h: usize) {
    let cx = 0.5 * (w as f64);
    let cy = 0.5 * (h as f64);
    let max_r = 0.355 * (w.min(h) as f64);
    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let d = ((x as f64) - cx).hypot((y as f64) - cy);
            if d <= max_r {
                let core_boost = 1.65 * (1.0 - 0.35 * (d / max_r));
                engine.trail_mut()[flatten_2d_index(x, y, w)] = core_boost;
            }
        }
    }
    let mut coords = Vec::with_capacity(MAX_PHYSARUM_TERMINALS);
    coords.extend_from_slice(engine.terminals());
    let k_len = coords.len().max(1);
    for idx in 0..super::MAX_PHYSARUM_AGENTS {
        let u1 = rng_unit_f64(engine.rng_mut());
        let u2 = rng_unit_f64(engine.rng_mut());
        let u3 = rng_unit_f64(engine.rng_mut());
        let origin = (idx % k_len) as u8;
        let target = coords[(idx + 1) % k_len];
        let rad = max_r * 0.85 * u1.sqrt();
        let ang = u2 * TAU;
        let ax = (cx + rad * ang.cos()).clamp(2.0, (w - 3) as f64);
        let ay = (cy + rad * ang.sin()).clamp(2.0, (h - 3) as f64);
        let dir = (target.y - ay).atan2(target.x - ax);
        let ah = (dir + (u3 - 0.5) * 0.6).rem_euclid(TAU);
        engine.set_agent_slot(idx, ax, ay, ah, 1.0, origin);
    }
    debug_assert!(max_r > 10.0);
    assert!(w >= 64);
}

#[verified_engine::verified]
fn seed_corridor_plasmodium(engine: &mut PhysarumEngine, w: usize, h: usize, base_trail: f64) {
    let mut open_cells: Vec<[usize; 2]> = Vec::with_capacity(w * h / 2);
    for y in 2..(h - 2) {
        for x in 2..(w - 2) {
            let cell = flatten_2d_index(x, y, w);
            if engine.obstacles()[cell] == 0 {
                open_cells.push([x, y]);
                engine.trail_mut()[cell] = base_trail;
            }
        }
    }
    assert!(!open_cells.is_empty());
    let mut coords = Vec::with_capacity(MAX_PHYSARUM_TERMINALS);
    coords.extend_from_slice(engine.terminals());
    let k_len = coords.len().max(1);
    for idx in 0..super::MAX_PHYSARUM_AGENTS {
        let origin = idx % k_len;
        let t = coords[origin];
        let u1 = rng_unit_f64(engine.rng_mut());
        let u2 = rng_unit_f64(engine.rng_mut());
        let ang = u2 * TAU;
        let r = 2.5 * u1.sqrt();
        let ax = (t.x + r * ang.cos()).clamp(2.0, (w - 3) as f64);
        let ay = (t.y + r * ang.sin()).clamp(2.0, (h - 3) as f64);
        engine.set_agent_slot(idx, ax, ay, ang, 1.0, origin as u8);
    }
    debug_assert!(k_len >= 2);
}

#[verified_engine::verified]
fn seed_tokyo_web(engine: &mut PhysarumEngine, w: usize, h: usize) {
    for y in 2..(h - 2) {
        for x in 2..(w - 2) {
            let cell = flatten_2d_index(x, y, w);
            if engine.obstacles()[cell] == 0 {
                engine.trail_mut()[cell] = 0.55;
            }
        }
    }
    let mut coords = Vec::with_capacity(MAX_PHYSARUM_TERMINALS);
    coords.extend_from_slice(engine.terminals());
    let k_len = coords.len().max(1);
    for idx in 0..super::MAX_PHYSARUM_AGENTS {
        let origin = idx % k_len;
        let target = coords[(idx * 3 + 1) % k_len];
        let t = coords[origin];
        let blend = rng_unit_f64(engine.rng_mut());
        let jitter = (rng_unit_f64(engine.rng_mut()) - 0.5) * 4.0;
        let mut ax = (t.x + blend * (target.x - t.x) + jitter).clamp(2.0, (w - 3) as f64);
        let mut ay = (t.y + blend * (target.y - t.y) + jitter).clamp(2.0, (h - 3) as f64);
        if engine.obstacles()[flatten_2d_index(ax as usize, ay as usize, w)] != 0 {
            ax = t.x;
            ay = t.y;
        }
        let ang = (target.y - ay).atan2(target.x - ax).rem_euclid(TAU);
        engine.set_agent_slot(idx, ax, ay, ang, 1.0, origin as u8);
    }
    assert!(k_len >= 2);
    debug_assert!(w >= 64);
}
