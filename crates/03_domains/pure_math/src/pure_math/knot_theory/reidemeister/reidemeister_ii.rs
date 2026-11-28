//! Validated Reidemeister II bigon insertion, detection, and strand splicing.

// [cite:knot_braid_topology]

use crate::pure_math::knot_theory::KnotError;
use crate::pure_math::knot_theory::diagram::{
    Crossing, CrossingSign, DiagramSegment, Endpoint, PlanarDiagram,
};

#[verified_engine::verified]
pub(super) fn insert_rii_on_free_loops(
    diagram: &mut PlanarDiagram,
    over_loop: usize,
    under_loop: usize,
) -> Result<(usize, usize), KnotError> {
    if over_loop >= diagram.free_loops || under_loop >= diagram.free_loops {
        return Err(KnotError::SegmentIndexOutOfBounds {
            index: over_loop.max(under_loop),
            count: diagram.free_loops,
        });
    }
    if over_loop == under_loop {
        diagram.free_loops -= 1;
        insert_single_loop_rii_dumbbell(diagram);
    } else {
        diagram.free_loops -= 2;
        insert_two_loop_rii_lens(diagram);
    }
    Ok((0, 1))
}

#[verified_engine::verified]
fn insert_single_loop_rii_dumbbell(diagram: &mut PlanarDiagram) {
    diagram.crossings.push(Crossing {
        sign: CrossingSign::Negative,
        slots: [1, 3, 3, 0],
        position: [-0.18, 0.0],
    });
    diagram.crossings.push(Crossing {
        sign: CrossingSign::Positive,
        slots: [2, 2, 1, 0],
        position: [0.18, 0.0],
    });
    for (tc, ts, hc, hs, wps) in [
        (0, 3, 1, 3, vec![[0.0, -0.12]]),
        (1, 2, 0, 0, vec![[0.0, 0.12]]),
        (1, 1, 1, 0, vec![[0.36, 0.22], [0.58, 0.0], [0.36, -0.22]]),
        (
            0,
            2,
            0,
            1,
            vec![[-0.36, -0.22], [-0.58, 0.0], [-0.36, 0.22]],
        ),
    ] {
        diagram.segments.push(DiagramSegment {
            tail: Endpoint {
                crossing: tc,
                slot: ts,
            },
            head: Endpoint {
                crossing: hc,
                slot: hs,
            },
            waypoints: wps,
        });
    }
}

#[verified_engine::verified]
fn insert_two_loop_rii_lens(diagram: &mut PlanarDiagram) {
    diagram.crossings.push(Crossing {
        sign: CrossingSign::Negative,
        slots: [3, 2, 1, 0],
        position: [-0.18, 0.0],
    });
    diagram.crossings.push(Crossing {
        sign: CrossingSign::Positive,
        slots: [1, 2, 3, 0],
        position: [0.18, 0.0],
    });
    for (tc, ts, hc, hs, wps) in [
        (0, 3, 1, 3, vec![[0.0, 0.12]]),
        (0, 2, 1, 0, vec![[0.0, -0.12]]),
        (
            1,
            1,
            0,
            1,
            vec![[0.36, -0.20], [0.0, -0.48], [-0.36, -0.20]],
        ),
        (1, 2, 0, 0, vec![[0.36, 0.20], [0.0, 0.48], [-0.36, 0.20]]),
    ] {
        diagram.segments.push(DiagramSegment {
            tail: Endpoint {
                crossing: tc,
                slot: ts,
            },
            head: Endpoint {
                crossing: hc,
                slot: hs,
            },
            waypoints: wps,
        });
    }
}

#[verified_engine::verified]
pub(super) fn insert_rii_bigon(
    diagram: &mut PlanarDiagram,
    over_seg: usize,
    under_seg: usize,
) -> Result<(usize, usize), KnotError> {
    for first_sign in [CrossingSign::Negative, CrossingSign::Positive] {
        for parallel in [true, false] {
            let mut candidate = diagram.clone();
            let pair = insert_rii_bigon_oriented(
                &mut candidate,
                over_seg,
                under_seg,
                first_sign,
                parallel,
            );
            if candidate.validate().is_ok() {
                *diagram = candidate;
                return Ok(pair);
            }
        }
    }
    Err(KnotError::ReidemeisterNotApplicable {
        move_kind: "R-II add",
        reason: "selected segments do not bound a common planar region",
    })
}

#[verified_engine::verified]
fn insert_rii_bigon_oriented(
    diagram: &mut PlanarDiagram,
    over_seg: usize,
    under_seg: usize,
    first_sign: CrossingSign,
    parallel: bool,
) -> (usize, usize) {
    let head_o = diagram.segments[over_seg].head;
    let head_u = diagram.segments[under_seg].head;
    let p_o = segment_midpoint(diagram, over_seg);
    let p_u = segment_midpoint(diagram, under_seg);
    let center = [0.5 * (p_o[0] + p_u[0]), 0.5 * (p_o[1] + p_u[1])];
    let second_sign = first_sign.flipped();
    let m = diagram.segments.len();
    let (c_a, c_b) = (diagram.crossings.len(), diagram.crossings.len() + 1);
    let (s_mid_o, s_mid_u, s_after_o, s_after_u) = (m, m + 1, m + 2, m + 3);

    let (ca_u_in, ca_u_out, cb_u_in, cb_u_out) = if parallel {
        (under_seg, s_mid_u, s_mid_u, s_after_u)
    } else {
        (s_mid_u, s_after_u, under_seg, s_mid_u)
    };
    let mut slots_a = [ca_u_in, 0, ca_u_out, 0];
    slots_a[first_sign.over_in_slot() as usize] = over_seg;
    slots_a[first_sign.over_out_slot() as usize] = s_mid_o;

    let mut slots_b = [cb_u_in, 0, cb_u_out, 0];
    slots_b[second_sign.over_in_slot() as usize] = s_mid_o;
    slots_b[second_sign.over_out_slot() as usize] = s_after_o;

    diagram.crossings.push(Crossing {
        sign: first_sign,
        slots: slots_a,
        position: [center[0] - 0.11, center[1] - 0.08],
    });
    diagram.crossings.push(Crossing {
        sign: second_sign,
        slots: slots_b,
        position: [center[0] + 0.11, center[1] + 0.08],
    });
    append_rii_bigon_segments(
        diagram,
        (over_seg, under_seg),
        (c_a, c_b),
        (first_sign, second_sign, parallel),
        (head_o, head_u),
        center,
    );
    diagram.crossings[head_o.crossing].slots[head_o.slot as usize] = s_after_o;
    diagram.crossings[head_u.crossing].slots[head_u.slot as usize] = s_after_u;
    (c_a, c_b)
}

#[verified_engine::verified]
fn append_rii_bigon_segments(
    diagram: &mut PlanarDiagram,
    segs: (usize, usize),
    pair: (usize, usize),
    orient: (CrossingSign, CrossingSign, bool),
    heads: (Endpoint, Endpoint),
    center: [f64; 2],
) {
    let (over_seg, under_seg) = segs;
    let (c_a, c_b) = pair;
    let (first_sign, second_sign, parallel) = orient;
    let (head_o, head_u) = heads;
    let (u_entry_c, u_exit_c) = if parallel { (c_a, c_b) } else { (c_b, c_a) };
    diagram.segments[over_seg].head = Endpoint {
        crossing: c_a,
        slot: first_sign.over_in_slot(),
    };
    diagram.segments[under_seg].head = Endpoint {
        crossing: u_entry_c,
        slot: 0,
    };
    for (tc, ts, hc, hs, wps) in [
        (
            c_a,
            first_sign.over_out_slot(),
            c_b,
            second_sign.over_in_slot(),
            vec![[center[0] - 0.07, center[1] + 0.07]],
        ),
        (
            u_entry_c,
            2,
            u_exit_c,
            0,
            vec![[center[0] + 0.07, center[1] - 0.07]],
        ),
        (
            c_b,
            second_sign.over_out_slot(),
            head_o.crossing,
            head_o.slot,
            Vec::new(),
        ),
        (u_exit_c, 2, head_u.crossing, head_u.slot, Vec::new()),
    ] {
        diagram.segments.push(DiagramSegment {
            tail: Endpoint {
                crossing: tc,
                slot: ts,
            },
            head: Endpoint {
                crossing: hc,
                slot: hs,
            },
            waypoints: wps,
        });
    }
}

#[verified_engine::verified]
pub(super) fn segment_midpoint(diagram: &PlanarDiagram, seg_id: usize) -> [f64; 2] {
    let seg = &diagram.segments[seg_id];
    if let Some(wp) = seg.waypoints.get(seg.waypoints.len() / 2) {
        return *wp;
    }
    let pt = diagram.crossings[seg.tail.crossing].position;
    let ph = diagram.crossings[seg.head.crossing].position;
    [0.5 * (pt[0] + ph[0]), 0.5 * (pt[1] + ph[1])]
}

#[verified_engine::verified]
pub(super) fn find_rii_bigon(
    diagram: &PlanarDiagram,
    c1: usize,
    c2: usize,
) -> Option<(usize, usize)> {
    if c1 == c2 {
        return None;
    }
    let (cr1, cr2) = (diagram.crossings.get(c1)?, diagram.crossings.get(c2)?);
    if cr1.sign == cr2.sign {
        return None;
    }
    for &slot_o1 in &[1_u8, 3_u8] {
        let s_over = cr1.slots[slot_o1 as usize];
        let Some(slot_o2) = [1_u8, 3_u8]
            .into_iter()
            .find(|&s| cr2.slots[s as usize] == s_over)
        else {
            continue;
        };
        for &slot_u1 in &[0_u8, 2_u8] {
            let s_under = cr1.slots[slot_u1 as usize];
            if s_under == s_over {
                continue;
            }
            let Some(slot_u2) = [0_u8, 2_u8]
                .into_iter()
                .find(|&s| cr2.slots[s as usize] == s_under)
            else {
                continue;
            };
            if ((slot_u1 + 4 - slot_o1) % 4) + ((slot_u2 + 4 - slot_o2) % 4) == 4 {
                return Some((s_over, s_under));
            }
        }
    }
    None
}

#[verified_engine::verified]
pub(super) fn splice_rii_strand(
    diagram: &mut PlanarDiagram,
    c1: usize,
    c2: usize,
    is_over: bool,
    s_mid: usize,
    drop_segs: &mut Vec<usize>,
) {
    let (in1, out1) = if is_over {
        (
            diagram.crossings[c1].over_in(),
            diagram.crossings[c1].over_out(),
        )
    } else {
        (
            diagram.crossings[c1].under_in(),
            diagram.crossings[c1].under_out(),
        )
    };
    let (in2, out2) = if is_over {
        (
            diagram.crossings[c2].over_in(),
            diagram.crossings[c2].over_out(),
        )
    } else {
        (
            diagram.crossings[c2].under_in(),
            diagram.crossings[c2].under_out(),
        )
    };
    let (s_ext_in, s_ext_out) = if out1 == s_mid {
        (in1, out2)
    } else {
        (in2, out1)
    };
    if s_ext_in == s_ext_out {
        diagram.free_loops += 1;
        drop_segs.push(s_ext_in);
    } else {
        let new_head = diagram.segments[s_ext_out].head;
        diagram.segments[s_ext_in].head = new_head;
        diagram.crossings[new_head.crossing].slots[new_head.slot as usize] = s_ext_in;
        drop_segs.push(s_ext_out);
    }
}
