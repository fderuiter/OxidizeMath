//! 6D Stampfli-Gähler dodecagonal ($D_{12}$) cut-and-project hyperlattice kernels.

// [cite:solid_state]

use super::{
    NON_SINGULAR_BIAS, ProjectedTile2D, QuasicrystalParams, QuasicrystalStats,
    QuasicrystalTileKind, QuasicrystalWorkspace, SILVER_RATIO, STAMPFLI_RATIO,
};
use std::f64::consts::{FRAC_PI_6, PI};

const SQRT_3: f64 = 1.732_050_807_568_877_2;

impl QuasicrystalWorkspace {
    #[verified_engine::verified]
    pub(super) fn project_stampfli(&mut self, p: &QuasicrystalParams) -> QuasicrystalStats {
        let a = p.edge_length;
        let r_norm = p.aperture_radius / a;
        let r_sq = p.aperture_radius * p.aperture_radius;
        let [wx, wy] = p.phason;
        let m1_min = ((-r_norm - 3.5 + wx) / SQRT_3).floor() as i32 - 1;
        let m1_max = ((r_norm + 3.5 + wx) / SQRT_3).ceil() as i32 + 1;
        let m2_min = ((-r_norm - 3.5 + wy) / SQRT_3).floor() as i32 - 1;
        let m2_max = ((r_norm + 3.5 + wy) / SQRT_3).ceil() as i32 + 1;
        let mut counts = [0_usize; 4];

        for m1 in m1_min..=m1_max {
            let px_mid = SQRT_3 * f64::from(m1) - wx;
            for m2 in m2_min..=m2_max {
                let py_mid = SQRT_3 * f64::from(m2) - wy;
                if px_mid * px_mid + py_mid * py_mid > (r_norm + 3.5) * (r_norm + 3.5) {
                    continue;
                }
                let m0_mid = 0.5 * (px_mid - wx - f64::from(m2));
                let m3_mid = 0.5 * (py_mid - wy - f64::from(m1));
                for m0 in (m0_mid.floor() as i32 - 2)..=(m0_mid.ceil() as i32 + 2) {
                    for m3 in (m3_mid.floor() as i32 - 2)..=(m3_mid.ceil() as i32 + 2) {
                        let n = [m1, m0 + m2, m1 + m3, m2, m3, -m0];
                        if let Some(perp) = stampfli_accept(&n, p.phason) {
                            let par = stampfli_parallel(&n, a);
                            if par[0] * par[0] + par[1] * par[1] <= r_sq {
                                self.push_vertex(par, perp, 0);
                            }
                            self.emit_stampfli_triangles(&n, p, r_sq, &mut counts);
                            self.emit_stampfli_squares(&n, p, r_sq, &mut counts);
                            self.emit_stampfli_shields(&n, p, r_sq, &mut counts);
                        }
                    }
                }
            }
        }
        let sq_f = counts[1].max(1) as f64;
        QuasicrystalStats {
            vertex_count: self.vertices.len(),
            tile_count: self.tiles.len(),
            primary_tile_count: counts[0],
            secondary_tile_count: counts[1],
            tertiary_tile_count: counts[2],
            flipped_tile_count: counts[3],
            empirical_ratio: (counts[0] + counts[2]) as f64 / sq_f,
            theoretical_ratio: STAMPFLI_RATIO,
            secondary_geometric_ratio: SILVER_RATIO,
        }
    }

    #[verified_engine::verified]
    fn emit_stampfli_triangles(
        &mut self,
        n: &[i32; 6],
        p: &QuasicrystalParams,
        r_sq: f64,
        counts: &mut [usize; 4],
    ) {
        let a = p.edge_length;
        for (s_a, s_b) in [(0, 2), (2, 4), (1, 3), (3, 5)] {
            let n1 = add6(n, &stampfli_step(s_a));
            let n2 = add6(n, &stampfli_step(s_b));
            if stampfli_accept(&n1, p.phason).is_none() || stampfli_accept(&n2, p.phason).is_none()
            {
                continue;
            }
            let (v0, v1, v2) = (
                stampfli_parallel(n, a),
                stampfli_parallel(&n1, a),
                stampfli_parallel(&n2, a),
            );
            let cx = (v0[0] + v1[0] + v2[0]) / 3.0;
            let cy = (v0[1] + v1[1] + v2[1]) / 3.0;
            if cx * cx + cy * cy <= r_sq {
                let flipped = !(stampfli_accept(n, p.reference_phason).is_some()
                    && stampfli_accept(&n1, p.reference_phason).is_some()
                    && stampfli_accept(&n2, p.reference_phason).is_some());
                counts[0] += 1;
                counts[3] += usize::from(flipped);
                self.push_tile(ProjectedTile2D {
                    kind: QuasicrystalTileKind::StampfliTriangle,
                    vertices: [v0, v1, v2, [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    vertex_count: 3,
                    phason_flipped: flipped,
                });
            }
        }
    }

    #[verified_engine::verified]
    fn emit_stampfli_squares(
        &mut self,
        n: &[i32; 6],
        p: &QuasicrystalParams,
        r_sq: f64,
        counts: &mut [usize; 4],
    ) {
        let a = p.edge_length;
        for (s_a, s_b) in [(0, 3), (1, 4), (2, 5)] {
            let n1 = add6(n, &stampfli_step(s_a));
            let n2 = add6(n, &stampfli_step(s_b));
            let n3 = add6(&n1, &stampfli_step(s_b));
            if stampfli_accept(&n1, p.phason).is_none()
                || stampfli_accept(&n2, p.phason).is_none()
                || stampfli_accept(&n3, p.phason).is_none()
            {
                continue;
            }
            let (v0, v3) = (stampfli_parallel(n, a), stampfli_parallel(&n3, a));
            if 0.25 * ((v0[0] + v3[0]).powi(2) + (v0[1] + v3[1]).powi(2)) <= r_sq {
                let flipped = !(stampfli_accept(n, p.reference_phason).is_some()
                    && stampfli_accept(&n1, p.reference_phason).is_some()
                    && stampfli_accept(&n2, p.reference_phason).is_some()
                    && stampfli_accept(&n3, p.reference_phason).is_some());
                counts[1] += 1;
                counts[3] += usize::from(flipped);
                self.push_tile(ProjectedTile2D {
                    kind: QuasicrystalTileKind::StampfliSquare,
                    vertices: [
                        v0,
                        stampfli_parallel(&n1, a),
                        v3,
                        stampfli_parallel(&n2, a),
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                    vertex_count: 4,
                    phason_flipped: flipped,
                });
            }
        }
    }

    #[verified_engine::verified]
    fn emit_stampfli_shields(
        &mut self,
        n: &[i32; 6],
        p: &QuasicrystalParams,
        r_sq: f64,
        counts: &mut [usize; 4],
    ) {
        for m in 0..4 {
            let Some(vs_n) = stampfli_shield_vertices(n, m, p.phason) else {
                continue;
            };
            let mut verts = [[0.0_f64; 2]; 6];
            let (mut cx, mut cy) = (0.0, 0.0);
            for (idx, vn) in vs_n.iter().enumerate() {
                verts[idx] = stampfli_parallel(vn, p.edge_length);
                cx += verts[idx][0];
                cy += verts[idx][1];
            }
            if (cx / 6.0).powi(2) + (cy / 6.0).powi(2) <= r_sq {
                let flipped = stampfli_shield_vertices(n, m, p.reference_phason).is_none();
                counts[2] += 1;
                counts[3] += usize::from(flipped);
                self.push_tile(ProjectedTile2D {
                    kind: QuasicrystalTileKind::StampfliShield,
                    vertices: verts,
                    vertex_count: 6,
                    phason_flipped: flipped,
                });
            }
        }
    }
}

#[verified_engine::verified]
fn stampfli_parallel(n: &[i32; 6], a: f64) -> [f64; 2] {
    let scale = a / SQRT_3;
    let (mut x, mut y) = (0.0, 0.0);
    for (j, &nj) in n.iter().enumerate() {
        let (s, c) = ((j as f64) * FRAC_PI_6).sin_cos();
        x += f64::from(nj) * c;
        y += f64::from(nj) * s;
    }
    [scale * x, scale * y]
}

#[verified_engine::verified]
fn stampfli_accept(n: &[i32; 6], phason: [f64; 2]) -> Option<[f64; 2]> {
    let inv_s3 = 1.0 / SQRT_3;
    let (mut yx, mut yy) = (
        -(phason[0] + NON_SINGULAR_BIAS[0]),
        -(phason[1] + NON_SINGULAR_BIAS[1]),
    );
    for (j, &nj) in n.iter().enumerate() {
        let (s, c) = (5.0 * (j as f64) * FRAC_PI_6).sin_cos();
        let nf = f64::from(nj) * inv_s3;
        yx += nf * c;
        yy += nf * s;
    }
    let apothem = (PI / 12.0).cos();
    for k in 0..6 {
        let (s, c) = (((k as f64) + 0.5) * FRAC_PI_6).sin_cos();
        if (yx * c + yy * s).abs() > apothem {
            return None;
        }
    }
    Some([yx, yy])
}

#[verified_engine::verified]
fn stampfli_step(m: usize) -> [i32; 6] {
    const BASE: [[i32; 6]; 6] = [
        [0, 1, 0, 0, 0, -1],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [-1, 0, 0, 0, 1, 0],
    ];
    let idx = m % 12;
    let s = BASE[idx % 6];
    if idx < 6 {
        s
    } else {
        [-s[0], -s[1], -s[2], -s[3], -s[4], -s[5]]
    }
}

#[verified_engine::verified]
fn add6(a: &[i32; 6], b: &[i32; 6]) -> [i32; 6] {
    [
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
        a[4] + b[4],
        a[5] + b[5],
    ]
}

#[verified_engine::verified]
fn stampfli_shield_vertices(n: &[i32; 6], m: usize, phason: [f64; 2]) -> Option<[[i32; 6]; 6]> {
    stampfli_accept(n, phason)?;
    let seq = [m, m + 1, m + 4, m + 5, m + 8];
    let mut vs = [*n; 6];
    let mut cur = *n;
    for (idx, &st) in seq.iter().enumerate() {
        cur = add6(&cur, &stampfli_step(st));
        stampfli_accept(&cur, phason)?;
        vs[idx + 1] = cur;
    }
    let p_a = add6(&vs[0], &add6(&stampfli_step(m), &stampfli_step(m + 3)));
    let p_b = add6(&vs[2], &add6(&stampfli_step(m + 4), &stampfli_step(m + 7)));
    let p_c = add6(&vs[4], &add6(&stampfli_step(m + 8), &stampfli_step(m + 11)));
    if stampfli_accept(&p_a, phason).is_some()
        || stampfli_accept(&p_b, phason).is_some()
        || stampfli_accept(&p_c, phason).is_some()
    {
        return None;
    }
    Some(vs)
}
