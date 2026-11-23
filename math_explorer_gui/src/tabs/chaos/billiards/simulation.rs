//! Zero-allocation ray-fan seeding, exact Euclidean collision stepping, and geometry helpers.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    BilliardGeometryKind, DynamicalBilliardsTool, RaySlot, BILLIARD_FRAME_BUDGET_MS,
    MAX_BIRKHOFF_POINTS, MAX_CHORD_VERTICES, MAX_RAYS,
};
use crate::web_io::PlatformInstant;
use math_explorer::physics::chaos::billiards::{
    BilliardCollision, BilliardRay, CircularBilliard, EllipticalBilliard, SinaiBilliard,
    StadiumBilliard,
};
use std::f64::consts::{PI, TAU};

const WARMUP_BOUNCES_ON_FIRE: usize = 96;

fn push_capped_birkhoff(
    buf: &mut Vec<[f64; 2]>,
    write_idx: &mut usize,
    cap: usize,
    pt: [f64; 2],
) {
    if !pt[0].is_finite() || !pt[1].is_finite() {
        return;
    }
    if buf.len() < cap {
        buf.push(pt);
    } else if cap > 0 {
        let idx = *write_idx % cap;
        buf[idx] = pt;
        *write_idx = (*write_idx + 1) % cap;
    }
}

fn push_sliding_vertex(buf: &mut Vec<[f64; 2]>, cap: usize, pt: [f64; 2]) {
    if !pt[0].is_finite() || !pt[1].is_finite() {
        return;
    }
    if buf.len() >= cap {
        buf.remove(0);
    }
    buf.push(pt);
}

impl DynamicalBilliardsTool {
    /// Reconstructs the concrete Euclidean table structs from current slider parameters.
    pub fn sync_table_kernels(&mut self) {
        self.circle = CircularBilliard::new(self.circle_radius);
        let a = self.ellipse_a.max(self.ellipse_b);
        let b = self.ellipse_b.min(self.ellipse_a);
        self.ellipse = EllipticalBilliard::new(a, b);
        self.stadium = StadiumBilliard::new(self.stadium_half_straight, self.stadium_radius);
        self.sinai = SinaiBilliard::new(self.sinai_half_side, self.sinai_scatterer_radius);
    }

    /// World-space bounding box `[x_min, x_max, y_min, y_max]` enclosing the active table.
    #[must_use]
    pub fn world_bounds(&self) -> [f64; 4] {
        match self.geometry {
            BilliardGeometryKind::Circle => {
                let r = self.circle.radius * 1.12;
                [-r, r, -r, r]
            }
            BilliardGeometryKind::Ellipse => {
                let a = self.ellipse.semi_major * 1.12;
                let b = self.ellipse.semi_minor * 1.12;
                let m = a.max(b);
                [-m, m, -m, m]
            }
            BilliardGeometryKind::Stadium => {
                let hx = (self.stadium.half_straight + self.stadium.radius) * 1.12;
                let hy = self.stadium.radius * 1.15;
                let m = hx.max(hy);
                [-m, m, -m, m]
            }
            BilliardGeometryKind::Sinai => {
                let s = self.sinai.half_side * 1.12;
                [-s, s, -s, s]
            }
        }
    }

    /// Checks whether `pos` lies strictly inside the valid free region of the active table.
    #[must_use]
    pub fn contains_interior_point(&self, pos: [f64; 2]) -> bool {
        match self.geometry {
            BilliardGeometryKind::Circle => self.circle.contains(pos),
            BilliardGeometryKind::Ellipse => self.ellipse.contains(pos),
            BilliardGeometryKind::Stadium => self.stadium.contains(pos),
            BilliardGeometryKind::Sinai => self.sinai.contains(pos),
        }
    }

    /// Default valid interior launch coordinate for `geometry`.
    #[must_use]
    pub fn default_launch_for(&self, geometry: BilliardGeometryKind) -> [f64; 2] {
        match geometry {
            BilliardGeometryKind::Circle => [0.45 * self.circle.radius, 0.0],
            BilliardGeometryKind::Ellipse => [0.30 * self.ellipse.semi_major, 0.12],
            BilliardGeometryKind::Stadium => [0.20, 0.15 * self.stadium.radius],
            BilliardGeometryKind::Sinai => {
                let mid = 0.5 * (self.sinai.scatterer_radius + self.sinai.half_side);
                [mid, 0.10]
            }
        }
    }

    /// Clamps all billiard parameters into valid physical ranges and syncs table kernels.
    pub fn clamp_billiard_parameters(&mut self) {
        if !self.launch_angle.is_finite() {
            self.launch_angle = 0.58;
        }
        self.launch_angle = self.launch_angle.clamp(-PI, PI);
        if !self.fan_spread.is_finite() {
            self.fan_spread = 0.24;
        }
        self.fan_spread = self.fan_spread.clamp(0.0, 1.2);
        self.fan_count = self.fan_count.clamp(1, MAX_RAYS);
        self.bounces_per_frame = self.bounces_per_frame.clamp(1, 64);

        if !self.circle_radius.is_finite() {
            self.circle_radius = 1.0;
        }
        self.circle_radius = self.circle_radius.clamp(0.3, 2.5);

        if !self.ellipse_a.is_finite() {
            self.ellipse_a = 1.35;
        }
        self.ellipse_a = self.ellipse_a.clamp(0.6, 2.5);
        if !self.ellipse_b.is_finite() {
            self.ellipse_b = 1.0;
        }
        self.ellipse_b = self.ellipse_b.clamp(0.3, self.ellipse_a);

        if !self.stadium_radius.is_finite() {
            self.stadium_radius = 0.75;
        }
        self.stadium_radius = self.stadium_radius.clamp(0.2, 1.5);
        if !self.stadium_half_straight.is_finite() {
            self.stadium_half_straight = 0.70;
        }
        self.stadium_half_straight = self.stadium_half_straight.clamp(0.05, 2.0);

        if !self.sinai_half_side.is_finite() {
            self.sinai_half_side = 1.10;
        }
        self.sinai_half_side = self.sinai_half_side.clamp(0.6, 2.0);
        if !self.sinai_scatterer_radius.is_finite() {
            self.sinai_scatterer_radius = 0.38;
        }
        self.sinai_scatterer_radius = self
            .sinai_scatterer_radius
            .clamp(0.08, (self.sinai_half_side * 0.75).min(0.9));

        self.sync_table_kernels();
        if !self.contains_interior_point(self.launch_pos) {
            self.launch_pos = self.default_launch_for(self.geometry);
        }
    }

    /// Switches to `geometry`, selecting a valid interior launch point and firing a fresh fan.
    pub fn switch_geometry(&mut self, geometry: BilliardGeometryKind) {
        self.geometry = geometry;
        self.sync_table_kernels();
        self.launch_pos = self.default_launch_for(geometry);
        self.last_status_note = None;
        self.rebuild_tables_and_fire_fan();
    }

    /// Syncs table geometry and fires a ray fan from `launch_pos` with warm-up bounces.
    pub fn rebuild_tables_and_fire_fan(&mut self) {
        self.clamp_billiard_parameters();
        let pos = self.launch_pos;
        let _ = self.fire_fan_at(pos);
    }

    /// Clears all active rays.
    pub fn clear_rays(&mut self) {
        self.rays.clear();
        self.state_rev = self.state_rev.wrapping_add(1);
    }

    /// Fires a ray fan centered at `pos` if `pos` lies inside the valid billiard interior.
    pub fn fire_fan_at(&mut self, pos: [f64; 2]) -> Result<usize, &'static str> {
        if !pos[0].is_finite() || !pos[1].is_finite() {
            return Err("Launch coordinates must be finite.");
        }
        if !self.contains_interior_point(pos) {
            return Err("Point lies outside the valid billiard interior.");
        }
        self.launch_pos = pos;
        self.rays.clear();
        let count = self.fan_count.clamp(1, MAX_RAYS);
        let spread = self.fan_spread.clamp(0.0, 1.2);
        for idx in 0..count {
            let offset = if count == 1 {
                0.0
            } else {
                let frac = (idx as f64) / ((count - 1) as f64) - 0.5;
                frac * spread
            };
            let angle = self.launch_angle + offset;
            if let Some(ray) = BilliardRay::from_angle(pos, angle) {
                let mut vertices = Vec::with_capacity(MAX_CHORD_VERTICES);
                vertices.push(pos);
                self.rays.push(RaySlot {
                    ray,
                    color_index: idx,
                    vertices,
                    birkhoff_points: Vec::with_capacity(256),
                    birkhoff_write_idx: 0,
                    bounce_count: 0,
                });
            }
        }
        self.advance_bounces(WARMUP_BOUNCES_ON_FIRE);
        self.state_rev = self.state_rev.wrapping_add(1);
        Ok(self.rays.len())
    }

    /// Appends a single ray launched from `pos` at `launch_angle` without clearing existing rays.
    pub fn seed_single_ray_at(&mut self, pos: [f64; 2]) -> Result<usize, &'static str> {
        if !pos[0].is_finite() || !pos[1].is_finite() {
            return Err("Launch coordinates must be finite.");
        }
        if !self.contains_interior_point(pos) {
            return Err("Point lies outside the valid billiard interior.");
        }
        self.launch_pos = pos;
        let ray = BilliardRay::from_angle(pos, self.launch_angle)
            .ok_or("Invalid ray launch angle.")?;
        if self.rays.len() >= MAX_RAYS {
            self.rays.remove(0);
            for (idx, existing) in self.rays.iter_mut().enumerate() {
                existing.color_index = idx;
            }
        }
        let color_index = self.rays.len();
        let mut slot = RaySlot {
            ray,
            color_index,
            vertices: vec![pos],
            birkhoff_points: Vec::with_capacity(256),
            birkhoff_write_idx: 0,
            bounce_count: 0,
        };
        for _ in 0..WARMUP_BOUNCES_ON_FIRE {
            if let Some(hit) = Self::collide_with_geometry(
                self.geometry,
                &self.circle,
                &self.ellipse,
                &self.stadium,
                &self.sinai,
                &mut slot.ray,
            ) {
                Self::record_slot_hit(&mut slot, &hit);
            } else {
                break;
            }
        }
        self.rays.push(slot);
        self.state_rev = self.state_rev.wrapping_add(1);
        Ok(self.rays.len() - 1)
    }

    /// Advances active rays within the 60 FPS [`BILLIARD_FRAME_BUDGET_MS`] wall-clock guard.
    pub fn step_frame_with_budget(&mut self) {
        let start = PlatformInstant::now();
        let target = self.bounces_per_frame.clamp(1, 64);
        for _ in 0..target {
            self.step_one_bounce_all();
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            if elapsed_ms >= BILLIARD_FRAME_BUDGET_MS {
                self.last_frame_ms = elapsed_ms;
                return;
            }
        }
        self.last_frame_ms = start.elapsed().as_secs_f64() * 1000.0;
    }

    /// Advances all active rays by `bounces` exact specular reflections.
    pub fn advance_bounces(&mut self, bounces: usize) {
        let bounded = bounces.min(4_096);
        for _ in 0..bounded {
            self.step_one_bounce_all();
        }
    }

    fn step_one_bounce_all(&mut self) {
        for slot in &mut self.rays {
            if let Some(hit) = Self::collide_with_geometry(
                self.geometry,
                &self.circle,
                &self.ellipse,
                &self.stadium,
                &self.sinai,
                &mut slot.ray,
            ) {
                Self::record_slot_hit(slot, &hit);
            }
        }
    }

    fn collide_with_geometry(
        geometry: BilliardGeometryKind,
        circle: &CircularBilliard,
        ellipse: &EllipticalBilliard,
        stadium: &StadiumBilliard,
        sinai: &SinaiBilliard,
        ray: &mut BilliardRay,
    ) -> Option<BilliardCollision> {
        match geometry {
            BilliardGeometryKind::Circle => circle.collide(ray),
            BilliardGeometryKind::Ellipse => ellipse.collide(ray),
            BilliardGeometryKind::Stadium => stadium.collide(ray),
            BilliardGeometryKind::Sinai => sinai.collide(ray),
        }
    }

    fn record_slot_hit(slot: &mut RaySlot, hit: &BilliardCollision) {
        slot.bounce_count += 1;
        push_sliding_vertex(&mut slot.vertices, MAX_CHORD_VERTICES, hit.hit_pos);
        push_capped_birkhoff(
            &mut slot.birkhoff_points,
            &mut slot.birkhoff_write_idx,
            MAX_BIRKHOFF_POINTS,
            [hit.arc_fraction, hit.sin_phi],
        );
    }

    /// Closed world-space polyline tracing the outer boundary of the active billiard table.
    #[must_use]
    pub fn outer_boundary_polyline(&self) -> Vec<[f64; 2]> {
        let samples = 128;
        match self.geometry {
            BilliardGeometryKind::Circle => (0..samples)
                .map(|i| {
                    let th = (i as f64 / samples as f64) * TAU;
                    [self.circle.radius * th.cos(), self.circle.radius * th.sin()]
                })
                .collect(),
            BilliardGeometryKind::Ellipse => (0..samples)
                .map(|i| {
                    let th = (i as f64 / samples as f64) * TAU;
                    [
                        self.ellipse.semi_major * th.cos(),
                        self.ellipse.semi_minor * th.sin(),
                    ]
                })
                .collect(),
            BilliardGeometryKind::Stadium => {
                let a = self.stadium.half_straight;
                let r = self.stadium.radius;
                let mut pts = Vec::with_capacity(96);
                pts.push([-a, -r]);
                pts.push([a, -r]);
                for i in 0..=40 {
                    let th = -0.5 * PI + (i as f64 / 40.0) * PI;
                    pts.push([a + r * th.cos(), r * th.sin()]);
                }
                pts.push([-a, r]);
                for i in 0..=40 {
                    let th = 0.5 * PI + (i as f64 / 40.0) * PI;
                    pts.push([-a + r * th.cos(), r * th.sin()]);
                }
                pts
            }
            BilliardGeometryKind::Sinai => {
                let s = self.sinai.half_side;
                vec![[-s, -s], [s, -s], [s, s], [-s, s]]
            }
        }
    }
}
