#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{FlowElement, PotentialFlowTool, SerializedFlowElement};
use egui_plot::PlotPoint;
use std::hash::{Hash as _, Hasher as _};
use std::sync::Arc;

#[derive(Default)]
pub(crate) struct ArrowFieldCache {
    key: Option<u64>,
    origins: Arc<Vec<PlotPoint>>,
    tips: Arc<Vec<PlotPoint>>,
    pub(crate) eval_count: usize,
}

fn canonical_bits(val: f64) -> u64 {
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

impl PotentialFlowTool {
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
        h.finish().max(1)
    }

    pub(crate) fn compute_arrow_field(&self) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        if !self.view_range.is_finite()
            || self.view_range <= 0.0
            || !(2..=math_commons::registry::MAX_GRID_SIZE).contains(&self.grid_size)
        {
            return (Vec::new(), Vec::new());
        }
        let step = (2.0 * self.view_range) / (self.grid_size as f64);
        if !step.is_finite() || step <= 0.0 {
            return (Vec::new(), Vec::new());
        }
        let min = -self.view_range;
        let cap = (self.grid_size + 1) * (self.grid_size + 1);
        let mut origins = Vec::with_capacity(cap);
        let mut tips = Vec::with_capacity(cap);

        for ix in 0..=self.grid_size {
            let x = min + (ix as f64) * step;
            for iy in 0..=self.grid_size {
                let y = min + (iy as f64) * step;
                let v = self.field.velocity(x, y);
                if !v.x.is_finite() || !v.y.is_finite() {
                    continue;
                }
                let magnitude = v.norm();
                if !magnitude.is_finite() || magnitude <= 1e-6 {
                    continue;
                }
                let arrow_len = (magnitude * 0.1).min(step * 0.8);
                if !arrow_len.is_finite() || arrow_len <= 0.0 {
                    continue;
                }
                let direction = v / magnitude;
                let end = [x + direction.x * arrow_len, y + direction.y * arrow_len];
                if end[0].is_finite() && end[1].is_finite() {
                    origins.push([x, y]);
                    tips.push(end);
                }
            }
        }
        (origins, tips)
    }

    pub(crate) fn ensure_cached_arrows(&mut self) -> (Arc<Vec<PlotPoint>>, Arc<Vec<PlotPoint>>) {
        let key = self.field_geometry_key();
        if self.cache.key != Some(key) {
            let (raw_origins, raw_tips) = self.compute_arrow_field();
            let origins = raw_origins
                .into_iter()
                .map(|[x, y]| PlotPoint::new(x, y))
                .collect();
            let tips = raw_tips
                .into_iter()
                .map(|[x, y]| PlotPoint::new(x, y))
                .collect();
            self.cache.origins = Arc::new(origins);
            self.cache.tips = Arc::new(tips);
            self.cache.key = Some(key);
            self.cache.eval_count += 1;
        }
        (
            Arc::clone(&self.cache.origins),
            Arc::clone(&self.cache.tips),
        )
    }
}
