//! Validated Reidemeister II bigon insertion, detection, and strand splicing.

// [cite:knot_braid_topology]

use crate::pure_math::knot_theory::diagram::{
    Crossing, CrossingSign, DiagramSegment, Endpoint, PlanarDiagram,
};

#[verified_engine::verified]
pub(super) fn insert_rii_bigon(
    diagram: &mut PlanarDiagram,
    over_seg: usize,
    under_seg: usize,
) -> (usize, usize) {
    let head_o = diagram.segments[over_seg].head;
    let head_u = diagram.segments[under_seg].head;
    let p_o = segment_midpoint(diagram, over_seg);
    let p_u = segment_midpoint(diagram, under_seg);
    let center = [0.5 * (p_o[0] + p_u[0]), 0.5 * (p_o[1] + p_u[1])];
    let m = diagram.segments.len();
    let (c_neg, c_pos) = (diagram.crossings.len(), diagram.crossings.len() + 1);
    let (s_mid_o, s_mid_u, s_after_o, s_after_u) = (m, m + 1, m + 2, m + 3);

    diagram.crossings.push(Crossing {
        sign: CrossingSign::Negative,
        slots: [under_seg, over_seg, s_mid_u, s_mid_o],
        position: [center[0] - 0.11, center[1] - 0.08],
    });
    diagram.crossings.push(Crossing {
        sign: CrossingSign::Positive,
        slots: [s_mid_u, s_after_o, s_after_u, s_mid_o],
        position: [center[0] + 0.11, center[1] + 0.08],
    });
    diagram.segments[over_seg].head = Endpoint {
        crossing: c_neg,
        slot: 1,
    };
    diagram.segments[under_seg].head = Endpoint {
        crossing: c_neg,
        slot: 0,
    };
    for (tc, ts, hc, hs, wps) in [
        (
            c_neg,
            3,
            c_pos,
            3,
            vec![[center[0] - 0.07, center[1] + 0.07]],
        ),
        (
            c_neg,
            2,
            c_pos,
            0,
            vec![[center[0] + 0.07, center[1] - 0.07]],
        ),
        (c_pos, 1, head_o.crossing, head_o.slot, Vec::new()),
        (c_pos, 2, head_u.crossing, head_u.slot, Vec::new()),
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
    diagram.crossings[head_o.crossing].slots[head_o.slot as usize] = s_after_o;
    diagram.crossings[head_u.crossing].slots[head_u.slot as usize] = s_after_u;
    (c_neg, c_pos)
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
