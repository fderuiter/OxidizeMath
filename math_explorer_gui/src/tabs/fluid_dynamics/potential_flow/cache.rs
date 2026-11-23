#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    ElementType, FlowElement, PotentialFlowTool, SerializedFlowElement, DYE_DT, DYE_MAX_SPEED,
};
use egui_plot::PlotPoint;
use math_explorer::nalgebra::Vector2;
use math_explorer::physics::fluid_dynamics::solver::{PotentialTracerSystem, TracerProbeReadout};
use math_explorer::physics::fluid_dynamics::types::FluidProperties;
use std::f64::consts::PI;
use std::hash::{Hash as _, Hasher as _};
use std::sync::Arc;

#[derive(Default)]
pub(crate) struct ContourCache {
    pub(crate) contour_key: Option<u64>,
    pub(crate) streamlines: Vec<Arc<Vec<PlotPoint>>>,
    pub(crate) stagnation: Arc<Vec<PlotPoint>>,
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

fn is_inside_cylinder(x: f64, y: f64, cyl: Option<(f64, f64, f64)>, ratio: f64) -> bool {
    cyl.is_some_and(|(cx, cy, r)| (x - cx).hypot(y - cy) < r * ratio)
}

impl PotentialFlowTool {
    pub(crate) fn active_probe_readout(&self) -> Option<TracerProbeReadout> {
        let probe = self
            .dye_probes
            .get(self.selected_probe)
            .or_else(|| self.dye_probes.first())?;
        let freestream = self
            .elements
            .iter()
            .filter(|e| e.element_type == ElementType::Uniform)
            .fold(Vector2::zeros(), |acc, e| {
                let rad = e.angle.to_radians();
                acc + Vector2::new(e.strength * rad.cos(), e.strength * rad.sin())
            })
            .norm()
            .max(1.0);
        let air = FluidProperties::air();
        let tracer = PotentialTracerSystem::new(
            &self.field,
            Vector2::new(probe.head[0], probe.head[1]),
            DYE_MAX_SPEED,
        );
        Some(tracer.evaluate_probe(&air, freestream, 101_325.0, DYE_DT))
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

    pub(crate) fn contour_geometry_key(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.contour_levels.hash(&mut h);
        self.contour_resolution.hash(&mut h);
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
        self.contour_geometry_key().hash(&mut h);
        (self.selected_element_type as u8).hash(&mut h);
        canonical_bits(self.new_element_strength).hash(&mut h);
        canonical_bits(self.new_element_x).hash(&mut h);
        canonical_bits(self.new_element_y).hash(&mut h);
        canonical_bits(self.new_element_angle).hash(&mut h);
        self.show_vectors.hash(&mut h);
        self.show_streamlines.hash(&mut h);
        self.show_dye.hash(&mut h);
        canonical_bits(self.dye_rake_x).hash(&mut h);
        canonical_bits(self.dye_rake_y).hash(&mut h);
        canonical_bits(self.dye_rake_span).hash(&mut h);
        self.dye_count.hash(&mut h);
        self.stamp_mode.hash(&mut h);
        h.finish().max(1)
    }

    pub(crate) fn aerodynamic_summary(&self) -> (f64, f64, f64, f64) {
        let (mut ux, mut uy, mut gamma, mut net_q) = (0.0, 0.0, 0.0, 0.0);
        for el in &self.elements {
            match el.element_type {
                ElementType::Uniform => {
                    let rad = el.angle.to_radians();
                    ux += el.strength * rad.cos();
                    uy += el.strength * rad.sin();
                }
                ElementType::Vortex => gamma += el.strength,
                ElementType::Source => net_q += el.strength,
                ElementType::Sink => net_q -= el.strength,
                ElementType::Doublet => {}
            }
        }
        let u_inf = ux.hypot(uy);
        let lift = -u_inf * gamma;
        let drag = if net_q.abs() <= 1e-6 {
            0.0
        } else {
            (u_inf * net_q).abs()
        };
        (u_inf, gamma, lift, drag)
    }

    pub(crate) fn inferred_cylinder(&self) -> Option<(f64, f64, f64)> {
        let (u_inf, _, _, _) = self.aerodynamic_summary();
        let doublet = self
            .elements
            .iter()
            .find(|e| e.element_type == ElementType::Doublet && e.strength > 1e-6)?;
        if u_inf <= 1e-6 {
            return None;
        }
        let r = (doublet.strength / (2.0 * PI * u_inf)).sqrt();
        (r.is_finite() && r > 0.05).then_some((doublet.x, doublet.y, r))
    }

    pub(crate) fn compute_exterior_stagnation(&self) -> Vec<PlotPoint> {
        let r = self.view_range.max(0.5);
        let cyl = self.inferred_cylinder();
        self.field
            .detect_stagnation_points((-r, r, -r, r), 40)
            .into_iter()
            .filter(|p| !is_inside_cylinder(p.x, p.y, cyl, 0.96))
            .map(|p| PlotPoint::new(p.x, p.y))
            .collect()
    }

    pub(crate) fn flow_observables(&self) -> Vec<(&'static str, f64)> {
        let (u_inf, gamma, lift, drag) = self.aerodynamic_summary();
        let r_cyl = self.inferred_cylinder().map_or(0.0, |(_, _, r)| r);
        let stag_count = if self.contour_cache.contour_key == Some(self.contour_geometry_key()) {
            self.contour_cache.stagnation.len()
        } else {
            self.compute_exterior_stagnation().len()
        };
        vec![
            ("freestream_speed", u_inf),
            ("circulation", gamma),
            ("circulation_abs", gamma.abs()),
            ("kutta_joukowski_lift", lift),
            ("lift_magnitude", lift.abs()),
            ("pressure_drag", drag),
            ("cylinder_radius", r_cyl),
            ("stagnation_count", stag_count as f64),
        ]
    }

    pub(super) fn populate_arrow_buffers(
        &self,
        origins: &mut Vec<PlotPoint>,
        tips: &mut Vec<PlotPoint>,
    ) {
        origins.clear();
        tips.clear();
        if !self.view_range.is_finite()
            || self.view_range <= 0.0
            || !(2..=math_commons::registry::MAX_GRID_SIZE).contains(&self.grid_size)
        {
            return;
        }
        let step = (2.0 * self.view_range) / (self.grid_size as f64);
        if !step.is_finite() || step <= 0.0 {
            return;
        }
        let target_cap = (self.grid_size + 1).pow(2);
        origins.reserve(target_cap);
        tips.reserve(target_cap);
        if self.elements.is_empty() {
            return;
        }
        let min = -self.view_range;
        let cyl = self.inferred_cylinder();
        for ix in 0..=self.grid_size {
            let x = min + (ix as f64) * step;
            for iy in 0..=self.grid_size {
                let y = min + (iy as f64) * step;
                if is_inside_cylinder(x, y, cyl, 0.92) {
                    continue;
                }
                let v = self.field.velocity(x, y);
                let mag = v.norm();
                if !v.x.is_finite() || !v.y.is_finite() || !mag.is_finite() || mag <= 1e-6 {
                    continue;
                }
                let arrow_len = (mag * 0.1).min(step * 0.8);
                let dir = v / mag;
                let end = [x + dir.x * arrow_len, y + dir.y * arrow_len];
                if arrow_len > 0.0 && end[0].is_finite() && end[1].is_finite() {
                    origins.push(PlotPoint::new(x, y));
                    tips.push(PlotPoint::new(end[0], end[1]));
                }
            }
        }
    }

    pub(crate) fn compute_arrow_field(&self) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        let mut origins = Vec::new();
        let mut tips = Vec::new();
        self.populate_arrow_buffers(&mut origins, &mut tips);
        (
            origins.into_iter().map(|p| [p.x, p.y]).collect(),
            tips.into_iter().map(|p| [p.x, p.y]).collect(),
        )
    }

    pub(crate) fn rebuild_arrows(&mut self) {
        let mut origins = std::mem::take(&mut self.origins);
        let mut tips = std::mem::take(&mut self.tips);
        self.populate_arrow_buffers(&mut origins, &mut tips);
        self.origins = origins;
        self.tips = tips;
        self.cached_grid_size = self.grid_size;
        self.cached_arrow_view_bits = canonical_bits(self.view_range);
        self.cached_revision = self.grid_revision;
    }

    pub(crate) fn reconcile_arrows(&mut self) {
        if self.cached_revision == self.grid_revision
            && (self.grid_size != self.cached_grid_size
                || canonical_bits(self.view_range) != self.cached_arrow_view_bits)
        {
            self.grid_revision = self.grid_revision.wrapping_add(1);
        }
        if self.cached_revision != self.grid_revision {
            self.rebuild_arrows();
        }
    }

    pub(crate) fn refresh_contour_cache(&mut self) {
        let key = self.contour_geometry_key();
        if self.contour_cache.contour_key == Some(key) {
            return;
        }
        let r = self.view_range.max(0.5);
        let bounds = (-r, r, -r, r);
        let levels = self.compute_iso_levels();
        let segs = self
            .field
            .extract_streamline_segments(bounds, self.contour_resolution, &levels);
        let chains = super::presets::chain_segments(&segs);
        self.contour_cache.streamlines = chains
            .iter()
            .map(|chain| Arc::new(chain.iter().map(|&[x, y]| PlotPoint::new(x, y)).collect()))
            .collect();
        self.cached_polylines = chains;
        self.contour_cache.stagnation = Arc::new(self.compute_exterior_stagnation());
        self.contour_cache.contour_key = Some(key);
    }

    pub(crate) fn ensure_cached_contours(
        &mut self,
    ) -> (Vec<Arc<Vec<PlotPoint>>>, Arc<Vec<PlotPoint>>) {
        self.refresh_contour_cache();
        (
            self.contour_cache
                .streamlines
                .iter()
                .map(Arc::clone)
                .collect(),
            Arc::clone(&self.contour_cache.stagnation),
        )
    }
}
