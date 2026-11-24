//! State serialization, streamline/stagnation contour caching, polyline chaining,
//! aerodynamic observables, and `RungeKutta4` dye advection.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    DyeStreak, ElementType, PotentialFlowTool, SerializedFlowElement, DYE_EMITTER_COUNT,
    DYE_SPEED_MAX, DYE_SPEED_MIN, DYE_TRAIL_CAP, FLOW_ANGLE_MAX, FLOW_ANGLE_MIN, FLOW_POS_MAX,
    FLOW_POS_MIN, FLOW_STRENGTH_MAX, FLOW_STRENGTH_MIN, FLOW_VIEW_MAX, FLOW_VIEW_MIN,
};
use math_explorer::physics::fluid_dynamics::potential_flow::FlowElement;
use math_explorer::physics::fluid_dynamics::solver::PotentialTracerSystem;
use math_explorer::pure_math::analysis::ode::TimeStepper;
use nalgebra::Vector2;
use std::collections::VecDeque;
use std::f64::consts::PI;
use std::hash::{Hash as _, Hasher as _};

const fn default_element_strength() -> f64 {
    10.0
}

const fn default_true() -> bool {
    true
}

const fn default_dye_speed() -> f64 {
    1.0
}

const fn default_probe_pos() -> [f64; 2] {
    [-4.0, 2.2]
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PotentialFlowState {
    elements: Vec<SerializedFlowElement>,
    grid_size: usize,
    view_range: f64,
    #[serde(default)]
    selected_element_type: ElementType,
    #[serde(default = "default_element_strength")]
    new_element_strength: f64,
    #[serde(default)]
    new_element_x: f64,
    #[serde(default)]
    new_element_y: f64,
    #[serde(default)]
    new_element_angle: f64,
    #[serde(default = "default_true")]
    show_arrows: bool,
    #[serde(default = "default_true")]
    show_streamlines: bool,
    #[serde(default = "default_true")]
    show_stagnation_points: bool,
    #[serde(default = "default_true")]
    show_dye_streaks: bool,
    #[serde(default = "default_true")]
    show_probe: bool,
    #[serde(default)]
    dye_playing: bool,
    #[serde(default = "default_dye_speed")]
    dye_speed: f64,
    #[serde(default = "default_probe_pos")]
    probe_pos: [f64; 2],
}

impl PotentialFlowState {
    fn is_valid(&self) -> bool {
        (2..=math_commons::registry::MAX_GRID_SIZE).contains(&self.grid_size)
            && self.view_range.is_finite()
            && (FLOW_VIEW_MIN..=FLOW_VIEW_MAX).contains(&self.view_range)
            && self.new_element_strength.is_finite()
            && (FLOW_STRENGTH_MIN..=FLOW_STRENGTH_MAX).contains(&self.new_element_strength)
            && self.new_element_x.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.new_element_x)
            && self.new_element_y.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.new_element_y)
            && self.new_element_angle.is_finite()
            && (FLOW_ANGLE_MIN..=FLOW_ANGLE_MAX).contains(&self.new_element_angle)
            && self.dye_speed.is_finite()
            && (DYE_SPEED_MIN..=DYE_SPEED_MAX).contains(&self.dye_speed)
            && self
                .probe_pos
                .iter()
                .all(|v| v.is_finite() && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(v))
            && self.elements.len() <= crate::framework::MAX_JSON_ARRAY_LEN
            && self.elements.iter().all(SerializedFlowElement::is_valid)
    }
}

pub(super) fn canonical_bits(val: f64) -> u64 {
    if val == 0.0 {
        0
    } else {
        val.to_bits()
    }
}

fn hash_flow_element(el: &SerializedFlowElement, h: &mut impl std::hash::Hasher) {
    (el.element_type as u8).hash(h);
    canonical_bits(el.strength).hash(h);
    canonical_bits(el.x).hash(h);
    canonical_bits(el.y).hash(h);
    canonical_bits(el.angle).hash(h);
}

fn chain_segments_into_polylines(segments: &[([f64; 2], [f64; 2])]) -> Vec<Vec<[f64; 2]>> {
    const SNAP: f64 = 1e-4;
    let mut used = vec![false; segments.len()];
    let mut polylines = Vec::new();
    let close = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() < SNAP && (a[1] - b[1]).abs() < SNAP;

    for start_idx in 0..segments.len() {
        if used[start_idx] {
            continue;
        }
        used[start_idx] = true;
        let mut chain = VecDeque::with_capacity(32);
        chain.push_back(segments[start_idx].0);
        chain.push_back(segments[start_idx].1);

        let mut extended = true;
        while extended {
            extended = false;
            let front = *chain.front().unwrap();
            let back = *chain.back().unwrap();
            for (idx, &(a, b)) in segments.iter().enumerate() {
                if used[idx] {
                    continue;
                }
                if close(back, a) {
                    chain.push_back(b);
                } else if close(back, b) {
                    chain.push_back(a);
                } else if close(front, b) {
                    chain.push_front(a);
                } else if close(front, a) {
                    chain.push_front(b);
                } else {
                    continue;
                }
                used[idx] = true;
                extended = true;
            }
        }
        polylines.push(chain.into_iter().collect());
    }
    polylines
}

impl PotentialFlowTool {
    pub(crate) fn save_state_json(&self) -> Option<String> {
        serde_json::to_string(&PotentialFlowState {
            elements: self.elements.clone(),
            grid_size: self.grid_size,
            view_range: self.view_range,
            selected_element_type: self.selected_element_type,
            new_element_strength: self.new_element_strength,
            new_element_x: self.new_element_x,
            new_element_y: self.new_element_y,
            new_element_angle: self.new_element_angle,
            show_arrows: self.show_arrows,
            show_streamlines: self.show_streamlines,
            show_stagnation_points: self.show_stagnation_points,
            show_dye_streaks: self.show_dye_streaks,
            show_probe: self.show_probe,
            dye_playing: self.dye_playing,
            dye_speed: self.dye_speed,
            probe_pos: self.probe_pos,
        })
        .ok()
    }

    pub(crate) fn load_state_json(&mut self, state: &str) -> Result<(), String> {
        let loaded: PotentialFlowState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid PotentialFlowTool state: {e}"))?;
        if !loaded.is_valid() {
            return Err(
                "PotentialFlowTool parameters must be finite and within valid grid/view bounds"
                    .to_string(),
            );
        }
        self.elements = loaded.elements;
        self.grid_size = loaded
            .grid_size
            .min(crate::presets::MAX_INTERACTIVE_GRID_SIZE);
        self.view_range = loaded.view_range;
        self.selected_element_type = loaded.selected_element_type;
        self.new_element_strength = loaded.new_element_strength;
        self.new_element_x = loaded.new_element_x;
        self.new_element_y = loaded.new_element_y;
        self.new_element_angle = loaded.new_element_angle;
        self.show_arrows = loaded.show_arrows;
        self.show_streamlines = loaded.show_streamlines;
        self.show_stagnation_points = loaded.show_stagnation_points;
        self.show_dye_streaks = loaded.show_dye_streaks;
        self.show_probe = loaded.show_probe;
        self.dye_playing = loaded.dye_playing;
        self.dye_speed = loaded.dye_speed;
        self.probe_pos = loaded.probe_pos;
        self.selected_element = None;
        self.dragged_element = None;
        self.active_drag_element = None;
        self.rebuild_field();
        self.reseed_dye_emitters();
        Ok(())
    }

    pub(crate) fn field_geometry_key(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.grid_size.hash(&mut h);
        canonical_bits(self.view_range).hash(&mut h);
        self.elements.len().hash(&mut h);
        for el in &self.elements {
            hash_flow_element(el, &mut h);
        }
        h.finish().max(1)
    }

    pub(crate) fn compute_state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.field_geometry_key().hash(&mut h);
        (self.selected_element_type as u8).hash(&mut h);
        canonical_bits(self.new_element_strength).hash(&mut h);
        canonical_bits(self.new_element_x).hash(&mut h);
        canonical_bits(self.new_element_y).hash(&mut h);
        canonical_bits(self.new_element_angle).hash(&mut h);
        self.show_arrows.hash(&mut h);
        self.show_streamlines.hash(&mut h);
        self.show_stagnation_points.hash(&mut h);
        self.show_dye_streaks.hash(&mut h);
        self.show_probe.hash(&mut h);
        self.dye_playing.hash(&mut h);
        canonical_bits(self.dye_speed).hash(&mut h);
        canonical_bits(self.probe_pos[0]).hash(&mut h);
        canonical_bits(self.probe_pos[1]).hash(&mut h);
        h.finish().max(1)
    }

    pub(crate) fn check_finite_parameters(&self) -> bool {
        (2..=crate::presets::MAX_INTERACTIVE_GRID_SIZE).contains(&self.grid_size)
            && self.view_range.is_finite()
            && (FLOW_VIEW_MIN..=FLOW_VIEW_MAX).contains(&self.view_range)
            && self.new_element_strength.is_finite()
            && (FLOW_STRENGTH_MIN..=FLOW_STRENGTH_MAX).contains(&self.new_element_strength)
            && self.new_element_x.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.new_element_x)
            && self.new_element_y.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.new_element_y)
            && self.new_element_angle.is_finite()
            && (FLOW_ANGLE_MIN..=FLOW_ANGLE_MAX).contains(&self.new_element_angle)
            && self.dye_speed.is_finite()
            && (DYE_SPEED_MIN..=DYE_SPEED_MAX).contains(&self.dye_speed)
            && self
                .probe_pos
                .iter()
                .all(|v| v.is_finite() && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(v))
            && self.elements.len() <= crate::framework::MAX_JSON_ARRAY_LEN
            && self.elements.iter().all(SerializedFlowElement::is_valid)
    }

    pub(crate) fn ensure_contour_cache(&mut self) {
        let key = self.field_geometry_key();
        if self.cached_field_key == key {
            return;
        }
        self.cached_field_key = key;
        if self.elements.is_empty() || !self.view_range.is_finite() || self.view_range <= 0.0 {
            self.cached_streamlines.clear();
            self.cached_polylines.clear();
            self.cached_stagnation.clear();
            return;
        }
        let r = self.view_range;
        let bounds = (-r, r, -r, r);
        let levels = self.sample_iso_levels(19);
        self.cached_streamlines = self.field.extract_streamline_segments(bounds, 56, &levels);
        self.cached_polylines = chain_segments_into_polylines(&self.cached_streamlines);
        self.cached_stagnation = self
            .field
            .detect_stagnation_points(bounds, 44)
            .into_iter()
            .filter(|p| !self.is_inside_dipole_cylinder(p.x, p.y))
            .map(|p| [p.x, p.y])
            .collect();
    }

    pub(crate) fn inferred_cylinder(&self) -> Option<(f64, f64, f64)> {
        let u = self.freestream_magnitude();
        if u <= 1e-6 {
            return None;
        }
        let doublet = self
            .elements
            .iter()
            .find(|e| e.element_type == ElementType::Doublet && e.strength.abs() > 1e-6)?;
        let radius = (doublet.strength.abs() / (2.0 * PI * u)).sqrt();
        (radius.is_finite() && radius > 1e-4).then_some((doublet.x, doublet.y, radius))
    }

    pub(crate) fn is_inside_dipole_cylinder(&self, x: f64, y: f64) -> bool {
        let Some((cx, cy, radius)) = self.inferred_cylinder() else {
            return false;
        };
        let dx = x - cx;
        let dy = y - cy;
        dx * dx + dy * dy < radius * radius * 0.92
    }

    pub(crate) fn freestream_magnitude(&self) -> f64 {
        let mut vx = 0.0;
        let mut vy = 0.0;
        for el in &self.elements {
            if el.element_type == ElementType::Uniform {
                let rad = el.angle.to_radians();
                vx += el.strength * rad.cos();
                vy += el.strength * rad.sin();
            }
        }
        vx.hypot(vy)
    }

    pub(crate) fn aerodynamic_summary(&self) -> (f64, f64, f64, f64) {
        let u_inf = self.freestream_magnitude();
        let mut gamma = 0.0;
        let mut net_q = 0.0;
        for el in &self.elements {
            match el.element_type {
                ElementType::Vortex => gamma += el.strength,
                ElementType::Source => net_q += el.strength,
                ElementType::Sink => net_q -= el.strength,
                ElementType::Uniform | ElementType::Doublet => {}
            }
        }
        let lift = -u_inf * gamma;
        let drag = if net_q.abs() <= 1e-6 {
            0.0
        } else {
            (u_inf * net_q).abs()
        };
        (u_inf, gamma, lift, drag)
    }

    pub(crate) fn compute_aerodynamic_observables(&self) -> Vec<(&'static str, f64)> {
        let (u_inf, circulation, lift, drag) = self.aerodynamic_summary();
        let cyl_r = self.inferred_cylinder().map_or(0.0, |(_, _, r)| r);
        let stag_count = if self.cached_field_key == self.field_geometry_key() {
            self.cached_stagnation.len() as f64
        } else {
            let vr = self.view_range;
            self.field
                .detect_stagnation_points((-vr, vr, -vr, vr), 44)
                .into_iter()
                .filter(|p| !self.is_inside_dipole_cylinder(p.x, p.y))
                .count() as f64
        };
        vec![
            ("freestream_speed", u_inf),
            ("circulation", circulation),
            ("circulation_abs", circulation.abs()),
            ("kutta_joukowski_lift", lift),
            ("lift_magnitude", lift.abs()),
            ("pressure_drag", drag),
            ("cylinder_radius", cyl_r),
            ("stagnation_count", stag_count),
        ]
    }

    fn sample_iso_levels(&self, count: usize) -> Vec<f64> {
        let r = self.view_range * 0.92;
        let mut vals = Vec::with_capacity(121);
        for j in 0..=10 {
            let y = -r + (2.0 * r) * (j as f64 / 10.0);
            for i in 0..=10 {
                let x = -r + (2.0 * r) * (i as f64 / 10.0);
                if !self.field.is_near_singularity(x, y, 0.08 * r * r + 1e-3) {
                    let psi = self.field.stream_function(x, y);
                    if psi.is_finite() {
                        vals.push(psi);
                    }
                }
            }
        }
        vals.sort_by(|a, b| a.total_cmp(b));
        if vals.len() < 2 {
            return vec![0.0];
        }
        let lo = vals[vals.len() / 12];
        let hi = vals[vals.len() * 11 / 12];
        if (hi - lo).abs() < 1e-6 {
            return vec![0.5 * (lo + hi)];
        }
        let mut levels: Vec<f64> = (0..count)
            .map(|k| lo + (hi - lo) * (k as f64 / (count - 1).max(1) as f64))
            .collect();
        if !levels.iter().any(|v| v.abs() < (hi - lo) * 0.02) {
            levels.push(0.0);
        }
        for pt in self
            .field
            .detect_stagnation_points((-r, r, -r, r), 36)
            .into_iter()
            .take(3)
        {
            let s_psi = self.field.stream_function(pt.x, pt.y);
            if s_psi.is_finite() && !levels.iter().any(|v| (v - s_psi).abs() < (hi - lo) * 0.015) {
                levels.push(s_psi);
            }
        }
        levels
    }

    pub(crate) fn reseed_dye_emitters(&mut self) {
        let r = self.view_range.clamp(FLOW_VIEW_MIN, FLOW_VIEW_MAX);
        let x0 = -0.92 * r;
        self.dye_streaks = (0..DYE_EMITTER_COUNT)
            .map(|k| {
                let frac = (k as f64 + 0.5) / DYE_EMITTER_COUNT as f64;
                let y0 = (-0.82 + 1.64 * frac) * r;
                let start = [x0, y0];
                let mut trail = VecDeque::with_capacity(DYE_TRAIL_CAP);
                trail.push_back(start);
                DyeStreak {
                    emitter: start,
                    head: start,
                    trail,
                }
            })
            .collect();
        if !self.elements.is_empty() {
            self.step_dye_tracers_burst(16, 0.025);
        }
    }

    fn step_dye_tracers_burst(&mut self, steps: usize, dt: f64) {
        for _ in 0..steps {
            self.step_dye_tracers(dt);
        }
    }

    pub(crate) fn step_dye_tracers(&mut self, dt: f64) {
        if self.elements.is_empty() || !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let r = self.view_range * 1.08;
        let max_speed = self.freestream_speed() * 6.0 + 30.0;
        for streak in &mut self.dye_streaks {
            let mut tracer = PotentialTracerSystem::new(
                &self.field,
                Vector2::new(streak.head[0], streak.head[1]),
                max_speed,
            );
            tracer.step(dt);
            let nx = tracer.position.x;
            let ny = tracer.position.y;
            let stalled = (nx - streak.head[0]).hypot(ny - streak.head[1]) < 1e-5 * dt;
            let out_of_bounds = !nx.is_finite() || !ny.is_finite() || nx.abs() > r || ny.abs() > r;
            let hit_core = self.field.is_near_singularity(nx, ny, 0.002 * r * r + 1e-4);
            if stalled || out_of_bounds || hit_core {
                streak.head = streak.emitter;
                streak.trail.clear();
                streak.trail.push_back(streak.emitter);
            } else {
                streak.head = [nx, ny];
                if streak.trail.len() >= DYE_TRAIL_CAP {
                    streak.trail.pop_front();
                }
                streak.trail.push_back(streak.head);
            }
        }
    }
}
