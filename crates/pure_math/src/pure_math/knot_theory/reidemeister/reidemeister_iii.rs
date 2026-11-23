//! Validated Reidemeister III triangle detection and strand-reversal slides.

// [cite:knot_braid_topology]

use crate::pure_math::knot_theory::KnotError;
use crate::pure_math::knot_theory::diagram::{Crossing, DiagramSegment, Endpoint, PlanarDiagram};

/// Identified Reidemeister-III triangle `(c_top_bot, c_top_mid, c_mid_bot)` on a [`PlanarDiagram`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ReidemeisterTriangle {
    /// Crossing between top and bottom strands.
    pub c_top_bot: usize,
    /// Crossing between top and middle strands.
    pub c_top_mid: usize,
    /// Crossing between middle and bottom strands.
    pub c_mid_bot: usize,
}

impl PlanarDiagram {
    /// Lists all valid **Reidemeister III** triangles currently available in `self`.
    #[must_use]
    #[verified_engine::verified]
    pub fn find_reidemeister_iii_triangles(&self) -> Vec<ReidemeisterTriangle> {
        let c = self.crossings.len();
        let mut out = Vec::new();
        for top_bot in 0..c {
            for top_mid in 0..c {
                if top_mid == top_bot {
                    continue;
                }
                for mid_bot in 0..c {
                    if mid_bot == top_bot || mid_bot == top_mid {
                        continue;
                    }
                    if match_riii_triangle(self, top_bot, top_mid, mid_bot).is_some() {
                        out.push(ReidemeisterTriangle {
                            c_top_bot: top_bot,
                            c_top_mid: top_mid,
                            c_mid_bot: mid_bot,
                        });
                    }
                }
            }
        }
        out
    }

    /// Executes a **Reidemeister III** triangle slide on `(c_top_bot, c_top_mid, c_mid_bot)`.
    ///
    /// # Errors
    /// Returns [`KnotError`] if the triple does not form a valid planar R-III triangle.
    #[verified_engine::verified]
    pub fn apply_reidemeister_iii(
        &mut self,
        c_top_bot: usize,
        c_top_mid: usize,
        c_mid_bot: usize,
    ) -> Result<(), KnotError> {
        let n = self.crossings.len();
        if c_top_bot >= n || c_top_mid >= n || c_mid_bot >= n {
            return Err(KnotError::CrossingIndexOutOfBounds {
                index: c_top_bot.max(c_top_mid).max(c_mid_bot),
                count: n,
            });
        }
        let Some((s_tm, s_mb, s_tb)) = match_riii_triangle(self, c_top_bot, c_top_mid, c_mid_bot)
        else {
            return Err(KnotError::ReidemeisterNotApplicable {
                move_kind: "R-III slide",
                reason: "selected crossing triple does not bound a valid R-III triangle",
            });
        };
        execute_riii_slide(self, c_top_bot, c_top_mid, c_mid_bot, s_tm, s_mb, s_tb);
        Ok(())
    }
}

#[verified_engine::verified]
fn match_riii_triangle(
    diagram: &PlanarDiagram,
    c_tb: usize,
    c_tm: usize,
    c_mb: usize,
) -> Option<(usize, usize, usize)> {
    let cr_tb = &diagram.crossings[c_tb];
    let cr_tm = &diagram.crossings[c_tm];
    let cr_mb = &diagram.crossings[c_mb];

    let (s_tm, slot_tb_o, slot_tm_o) = find_shared_slot(cr_tb, &[1, 3], cr_tm, &[1, 3])?;
    let (s_mb, slot_tm_u, slot_mb_o) = find_shared_slot(cr_tm, &[0, 2], cr_mb, &[1, 3])?;
    let (s_tb, slot_mb_u, slot_tb_u) = find_shared_slot(cr_mb, &[0, 2], cr_tb, &[0, 2])?;

    let step_tb = (slot_tb_o + 4 - slot_tb_u) % 4;
    let step_tm = (slot_tm_u + 4 - slot_tm_o) % 4;
    let step_mb = (slot_mb_u + 4 - slot_mb_o) % 4;
    if (step_tb == 1 || step_tb == 3) && step_tb == step_tm && step_tm == step_mb {
        Some((s_tm, s_mb, s_tb))
    } else {
        None
    }
}

#[verified_engine::verified]
fn find_shared_slot(
    ca: &Crossing,
    slots_a: &[u8],
    cb: &Crossing,
    slots_b: &[u8],
) -> Option<(usize, u8, u8)> {
    for &sa in slots_a {
        let seg_a = ca.slots[sa as usize];
        for &sb in slots_b {
            if cb.slots[sb as usize] == seg_a {
                return Some((seg_a, sa, sb));
            }
        }
    }
    None
}

#[verified_engine::verified]
fn execute_riii_slide(
    diagram: &mut PlanarDiagram,
    c_tb: usize,
    c_tm: usize,
    c_mb: usize,
    s_tm: usize,
    s_mb: usize,
    s_tb: usize,
) {
    let (top_1, top_2) = if diagram.crossings[c_tb].over_out() == s_tm {
        (c_tb, c_tm)
    } else {
        (c_tm, c_tb)
    };
    let (mid_1, mid_2) = if diagram.crossings[c_tm].under_out() == s_mb {
        (c_tm, c_mb)
    } else {
        (c_mb, c_tm)
    };
    let (bot_1, bot_2) = if diagram.crossings[c_tb].under_out() == s_tb {
        (c_tb, c_mb)
    } else {
        (c_mb, c_tb)
    };

    let (ext_top_in, ext_top_out) = (
        diagram.crossings[top_1].over_in(),
        diagram.crossings[top_2].over_out(),
    );
    let ext_mid_in = if mid_1 == c_tm {
        diagram.crossings[c_tm].under_in()
    } else {
        diagram.crossings[c_mb].over_in()
    };
    let ext_mid_out = if mid_2 == c_tm {
        diagram.crossings[c_tm].under_out()
    } else {
        diagram.crossings[c_mb].over_out()
    };
    let (ext_bot_in, ext_bot_out) = (
        diagram.crossings[bot_1].under_in(),
        diagram.crossings[bot_2].under_out(),
    );

    rewire_strand_pair(
        diagram,
        (ext_top_in, ext_top_out, s_tm),
        (
            top_2,
            diagram.crossings[top_2].sign.over_in_slot(),
            diagram.crossings[top_2].sign.over_out_slot(),
        ),
        (
            top_1,
            diagram.crossings[top_1].sign.over_in_slot(),
            diagram.crossings[top_1].sign.over_out_slot(),
        ),
    );
    let m2_slots = mid_slots(diagram, mid_2, c_tm, c_mb);
    let m1_slots = mid_slots(diagram, mid_1, c_tm, c_mb);
    rewire_strand_pair(diagram, (ext_mid_in, ext_mid_out, s_mb), m2_slots, m1_slots);
    rewire_strand_pair(
        diagram,
        (ext_bot_in, ext_bot_out, s_tb),
        (bot_2, 0, 2),
        (bot_1, 0, 2),
    );
    reflect_triangle_positions(diagram, [c_tb, c_tm, c_mb]);
}

#[verified_engine::verified]
fn mid_slots(diagram: &PlanarDiagram, c: usize, c_tm: usize, c_mb: usize) -> (usize, u8, u8) {
    if c == c_tm {
        (c_tm, 0, 2)
    } else {
        let sign = diagram.crossings[c_mb].sign;
        (c_mb, sign.over_in_slot(), sign.over_out_slot())
    }
}

#[verified_engine::verified]
fn rewire_strand_pair(
    diagram: &mut PlanarDiagram,
    segs: (usize, usize, usize),
    first_new: (usize, u8, u8),
    second_new: (usize, u8, u8),
) {
    let (ext_in, ext_out, s_int) = segs;
    diagram.segments[ext_in].head = Endpoint {
        crossing: first_new.0,
        slot: first_new.1,
    };
    diagram.crossings[first_new.0].slots[first_new.1 as usize] = ext_in;

    diagram.segments[ext_out].tail = Endpoint {
        crossing: second_new.0,
        slot: second_new.2,
    };
    diagram.crossings[second_new.0].slots[second_new.2 as usize] = ext_out;

    diagram.segments[s_int] = DiagramSegment {
        tail: Endpoint {
            crossing: first_new.0,
            slot: first_new.2,
        },
        head: Endpoint {
            crossing: second_new.0,
            slot: second_new.1,
        },
        waypoints: Vec::new(),
    };
    diagram.crossings[first_new.0].slots[first_new.2 as usize] = s_int;
    diagram.crossings[second_new.0].slots[second_new.1 as usize] = s_int;
}

#[verified_engine::verified]
fn reflect_triangle_positions(diagram: &mut PlanarDiagram, ids: [usize; 3]) {
    let pa = diagram.crossings[ids[0]].position;
    let pb = diagram.crossings[ids[1]].position;
    let pc = diagram.crossings[ids[2]].position;
    let cx = (pa[0] + pb[0] + pc[0]) / 3.0;
    let cy = (pa[1] + pb[1] + pc[1]) / 3.0;
    for cid in ids {
        let p = diagram.crossings[cid].position;
        diagram.crossings[cid].position = [2.0 * cx - p[0], 2.0 * cy - p[1]];
    }
}
