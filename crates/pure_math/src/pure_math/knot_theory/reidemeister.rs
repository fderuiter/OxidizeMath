//! Validated Reidemeister I and II ambient isotopy moves on [`PlanarDiagram`].

// [cite:knot_braid_topology]

use super::KnotError;
use super::diagram::{
    Crossing, CrossingSign, DiagramSegment, Endpoint, MAX_DIAGRAM_CROSSINGS, PlanarDiagram,
};

mod reidemeister_ii;
mod reidemeister_iii;
use self::reidemeister_ii::{
    find_rii_bigon, insert_rii_bigon, segment_midpoint, splice_rii_strand,
};
pub use reidemeister_iii::ReidemeisterTriangle;

impl PlanarDiagram {
    /// Applies a **Reidemeister I** twist addition (`sign`) on `segment_id` (or on `free_loops`
    /// when `crossings.is_empty() && segment_id == 0`), changing diagram writhe by $\pm 1$.
    ///
    /// # Errors
    /// Returns [`KnotError`] if `segment_id` is out of bounds or the crossing cap is exceeded.
    #[verified_engine::verified]
    pub fn apply_reidemeister_i_add(
        &mut self,
        segment_id: usize,
        sign: CrossingSign,
    ) -> Result<usize, KnotError> {
        if self.crossings.len() + 1 > MAX_DIAGRAM_CROSSINGS {
            return Err(KnotError::InvalidDiagram {
                reason: "Reidemeister I add would exceed maximum crossing bound",
            });
        }
        if self.crossings.is_empty() {
            return add_ri_on_free_loop(self, segment_id, sign);
        }
        if segment_id >= self.segments.len() {
            return Err(KnotError::SegmentIndexOutOfBounds {
                index: segment_id,
                count: self.segments.len(),
            });
        }
        Ok(insert_ri_curl_on_segment(self, segment_id, sign))
    }

    /// Returns `true` if `crossing_id` bounds a removable Reidemeister-I monogon curl.
    #[must_use]
    #[verified_engine::verified]
    pub fn can_apply_reidemeister_i_remove(&self, crossing_id: usize) -> bool {
        find_ri_loop_segment(self, crossing_id).is_some()
    }

    /// Untwists a removable **Reidemeister I** monogon at `crossing_id`.
    ///
    /// # Errors
    /// Returns [`KnotError`] if `crossing_id` is out of bounds or not a Reidemeister-I curl.
    #[verified_engine::verified]
    pub fn apply_reidemeister_i_remove(&mut self, crossing_id: usize) -> Result<(), KnotError> {
        if crossing_id >= self.crossings.len() {
            return Err(KnotError::CrossingIndexOutOfBounds {
                index: crossing_id,
                count: self.crossings.len(),
            });
        }
        let Some((s_loop, s_in, s_out)) = find_ri_loop_segment(self, crossing_id) else {
            return Err(KnotError::ReidemeisterNotApplicable {
                move_kind: "R-I remove",
                reason: "selected crossing has no adjacent monogon curl",
            });
        };
        if s_in == s_out {
            self.free_loops += 1;
            compact_remove(self, &[crossing_id], &[s_loop, s_in]);
            return Ok(());
        }
        let new_head = self.segments[s_out].head;
        let extra_wps = self.segments[s_out].waypoints.clone();
        self.segments[s_in].head = new_head;
        if self.segments[s_in].waypoints.len() + extra_wps.len() <= 16 {
            self.segments[s_in].waypoints.extend(extra_wps);
        }
        self.crossings[new_head.crossing].slots[new_head.slot as usize] = s_in;
        compact_remove(self, &[crossing_id], &[s_loop, s_out]);
        Ok(())
    }

    /// Applies a **Reidemeister II** bigon poke sliding `over_segment_id` across `under_segment_id`,
    /// inserting two opposite-sign crossings with zero net writhe change.
    ///
    /// # Errors
    /// Returns [`KnotError`] if segment indices are invalid, identical (on $>0$ crossings), or exceed bounds.
    #[verified_engine::verified]
    pub fn apply_reidemeister_ii_add(
        &mut self,
        over_segment_id: usize,
        under_segment_id: usize,
    ) -> Result<(usize, usize), KnotError> {
        if self.crossings.len() + 2 > MAX_DIAGRAM_CROSSINGS {
            return Err(KnotError::InvalidDiagram {
                reason: "Reidemeister II add would exceed maximum crossing bound",
            });
        }
        if self.crossings.is_empty() {
            if self.free_loops == 0 || over_segment_id != 0 || under_segment_id != 0 {
                return Err(KnotError::SegmentIndexOutOfBounds {
                    index: over_segment_id.max(under_segment_id),
                    count: 0,
                });
            }
            let _ = self.apply_reidemeister_i_add(0, CrossingSign::Positive)?;
            return Ok(insert_rii_bigon(self, 0, 1));
        }
        let m = self.segments.len();
        if over_segment_id >= m || under_segment_id >= m {
            return Err(KnotError::SegmentIndexOutOfBounds {
                index: over_segment_id.max(under_segment_id),
                count: m,
            });
        }
        if over_segment_id == under_segment_id {
            return Err(KnotError::ReidemeisterNotApplicable {
                move_kind: "R-II add",
                reason: "over and under segments must be distinct",
            });
        }
        Ok(insert_rii_bigon(self, over_segment_id, under_segment_id))
    }

    /// Returns `true` if `(c1, c2)` bound a removable Reidemeister-II bigon.
    #[must_use]
    #[verified_engine::verified]
    pub fn can_apply_reidemeister_ii_remove(&self, c1: usize, c2: usize) -> bool {
        find_rii_bigon(self, c1, c2).is_some()
    }

    /// Cancels a removable **Reidemeister II** bigon between crossings `c1` and `c2`.
    ///
    /// # Errors
    /// Returns [`KnotError`] if `(c1, c2)` do not form an untwisted opposite-sign bigon.
    #[verified_engine::verified]
    pub fn apply_reidemeister_ii_remove(&mut self, c1: usize, c2: usize) -> Result<(), KnotError> {
        let n = self.crossings.len();
        if c1 >= n || c2 >= n {
            return Err(KnotError::CrossingIndexOutOfBounds {
                index: c1.max(c2),
                count: n,
            });
        }
        let Some((s_over_mid, s_under_mid)) = find_rii_bigon(self, c1, c2) else {
            return Err(KnotError::ReidemeisterNotApplicable {
                move_kind: "R-II remove",
                reason: "selected crossings do not bound a removable opposite-sign bigon",
            });
        };
        let mut drop_segs = vec![s_over_mid, s_under_mid];
        splice_rii_strand(self, c1, c2, true, s_over_mid, &mut drop_segs);
        splice_rii_strand(self, c1, c2, false, s_under_mid, &mut drop_segs);
        compact_remove(self, &[c1, c2], &drop_segs);
        Ok(())
    }
}

#[verified_engine::verified]
fn add_ri_on_free_loop(
    diagram: &mut PlanarDiagram,
    segment_id: usize,
    sign: CrossingSign,
) -> Result<usize, KnotError> {
    if diagram.free_loops == 0 || segment_id != 0 {
        return Err(KnotError::SegmentIndexOutOfBounds {
            index: segment_id,
            count: 0,
        });
    }
    diagram.free_loops -= 1;
    let mut slots = [0; 4];
    slots[0] = 0;
    slots[sign.over_out_slot() as usize] = 0;
    slots[2] = 1;
    slots[sign.over_in_slot() as usize] = 1;
    diagram.crossings.push(Crossing {
        sign,
        slots,
        position: [0.0, 0.0],
    });
    let s = sign.value() as f64;
    diagram.segments.push(DiagramSegment {
        tail: Endpoint {
            crossing: 0,
            slot: sign.over_out_slot(),
        },
        head: Endpoint {
            crossing: 0,
            slot: 0,
        },
        waypoints: vec![
            [s * 0.14, -0.10],
            [s * 0.42, -0.26],
            [0.0, -0.46],
            [-0.12, -0.22],
        ],
    });
    diagram.segments.push(DiagramSegment {
        tail: Endpoint {
            crossing: 0,
            slot: 2,
        },
        head: Endpoint {
            crossing: 0,
            slot: sign.over_in_slot(),
        },
        waypoints: vec![
            [0.10, 0.16],
            [0.0, 0.46],
            [-s * 0.42, 0.26],
            [-s * 0.14, 0.08],
        ],
    });
    Ok(0)
}

#[verified_engine::verified]
fn insert_ri_curl_on_segment(
    diagram: &mut PlanarDiagram,
    segment_id: usize,
    sign: CrossingSign,
) -> usize {
    let orig_tail = diagram.segments[segment_id].tail;
    let orig_head = diagram.segments[segment_id].head;
    let p_tail = diagram.crossings[orig_tail.crossing].position;
    let p_head = diagram.crossings[orig_head.crossing].position;
    let mid = segment_midpoint(diagram, segment_id);
    let chord = [p_head[0] - p_tail[0], p_head[1] - p_tail[1]];
    let len = (chord[0] * chord[0] + chord[1] * chord[1]).sqrt().max(1e-3);
    let (tx, ty) = (chord[0] / len, chord[1] / len);
    let (lx, ly) = (-ty, tx);
    let s = sign.value() as f64;
    // Under-strand exits along +t, loops via +s*l, and re-enters over-strand from +s*l -> -s*l
    let curl_wps = vec![
        [mid[0] + 0.11 * tx, mid[1] + 0.11 * ty],
        [
            mid[0] + 0.18 * tx + s * 0.14 * lx,
            mid[1] + 0.18 * ty + s * 0.14 * ly,
        ],
        [mid[0] + s * 0.22 * lx, mid[1] + s * 0.22 * ly],
        [
            mid[0] - 0.04 * tx + s * 0.11 * lx,
            mid[1] - 0.04 * ty + s * 0.11 * ly,
        ],
    ];
    let c_new = diagram.crossings.len();
    let s_curl = diagram.segments.len();
    let s_after = s_curl + 1;
    let mut slots = [0; 4];
    slots[0] = segment_id;
    slots[2] = s_curl;
    slots[sign.over_in_slot() as usize] = s_curl;
    slots[sign.over_out_slot() as usize] = s_after;
    diagram.crossings.push(Crossing {
        sign,
        slots,
        position: mid,
    });
    let mut after_wps = vec![[
        mid[0] + 0.04 * tx - s * 0.11 * lx,
        mid[1] + 0.04 * ty - s * 0.11 * ly,
    ]];
    if diagram.segments[segment_id].waypoints.len() > 1 {
        after_wps.extend(diagram.segments[segment_id].waypoints.split_off(1));
    }
    diagram.segments[segment_id]
        .waypoints
        .push([mid[0] - 0.09 * tx, mid[1] - 0.09 * ty]);
    diagram.segments[segment_id].head = Endpoint {
        crossing: c_new,
        slot: 0,
    };
    diagram.segments.push(DiagramSegment {
        tail: Endpoint {
            crossing: c_new,
            slot: 2,
        },
        head: Endpoint {
            crossing: c_new,
            slot: sign.over_in_slot(),
        },
        waypoints: curl_wps,
    });
    diagram.segments.push(DiagramSegment {
        tail: Endpoint {
            crossing: c_new,
            slot: sign.over_out_slot(),
        },
        head: orig_head,
        waypoints: after_wps,
    });
    diagram.crossings[orig_head.crossing].slots[orig_head.slot as usize] = s_after;
    c_new
}

#[verified_engine::verified]
fn find_ri_loop_segment(
    diagram: &PlanarDiagram,
    crossing_id: usize,
) -> Option<(usize, usize, usize)> {
    let c = diagram.crossings.get(crossing_id)?;
    for slot_a in 0..4_u8 {
        let slot_b = (slot_a + 1) % 4;
        let seg_a = c.slots[slot_a as usize];
        if seg_a == c.slots[slot_b as usize] {
            let slot_c = (slot_a + 2) % 4;
            let slot_d = (slot_a + 3) % 4;
            let (s_in, s_out) = if c.sign.is_outgoing_slot(slot_c) {
                (c.slots[slot_d as usize], c.slots[slot_c as usize])
            } else {
                (c.slots[slot_c as usize], c.slots[slot_d as usize])
            };
            return Some((seg_a, s_in, s_out));
        }
    }
    None
}

#[verified_engine::verified]
pub(super) fn compact_remove(diagram: &mut PlanarDiagram, drop_c: &[usize], drop_s: &[usize]) {
    let mut c_map = vec![usize::MAX; diagram.crossings.len()];
    let mut new_crossings = Vec::with_capacity(diagram.crossings.len());
    for (idx, cr) in diagram.crossings.iter().enumerate() {
        if !drop_c.contains(&idx) {
            c_map[idx] = new_crossings.len();
            new_crossings.push(*cr);
        }
    }
    let mut s_map = vec![usize::MAX; diagram.segments.len()];
    let mut new_segments = Vec::with_capacity(diagram.segments.len());
    for (idx, seg) in diagram.segments.iter().enumerate() {
        if !drop_s.contains(&idx) {
            s_map[idx] = new_segments.len();
            let mut updated = seg.clone();
            updated.tail.crossing = c_map[updated.tail.crossing];
            updated.head.crossing = c_map[updated.head.crossing];
            new_segments.push(updated);
        }
    }
    for cr in &mut new_crossings {
        for slot in &mut cr.slots {
            *slot = s_map[*slot];
        }
    }
    diagram.crossings = new_crossings;
    diagram.segments = new_segments;
}
