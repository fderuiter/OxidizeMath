#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{PotentialFlowTool, SerializedFlowElement};
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
