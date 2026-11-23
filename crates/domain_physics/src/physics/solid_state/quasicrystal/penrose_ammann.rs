//! 5D Penrose ($D_{10}$) and 4D Ammann-Beenker ($D_8$) hyperlattice cut-and-project kernels.

// [cite:solid_state]

use super::{
    GOLDEN_RATIO, NON_SINGULAR_BIAS, PENROSE_GAMMA0, ProjectedTile2D, QuasicrystalParams,
    QuasicrystalStats, QuasicrystalTileKind, QuasicrystalWorkspace, SILVER_RATIO,
};
use std::f64::consts::{FRAC_PI_4, PI, SQRT_2};

impl QuasicrystalWorkspace {
    #[verified_engine::verified]
    pub(super) fn project_penrose(&mut self, p: &QuasicrystalParams) -> QuasicrystalStats {
        let r_norm = p.aperture_radius / p.edge_length;
        let r_sq = p.aperture_radius * p.aperture_radius;
        let [wx, wy] = p.phason;
        let (s1, c1) = (2.0 * PI / 5.0).sin_cos();
        let (s2, c2) = (4.0 * PI / 5.0).sin_cos();
        let mut counts = [0_usize; 4];

        for h in 1_i32..=5 {
            let hf = f64::from(h);
            let n0_min = ((0.2 * hf + 0.4 * (-r_norm - 4.0 + wx)).floor() as i32) - 1;
            let n0_max = ((0.2 * hf + 0.4 * (r_norm + 4.0 + wx)).ceil() as i32) + 1;
            let n1_center = 0.2 * hf + 0.4 * (wx * c2 + wy * s2);
            let n1_span = (0.4 * (r_norm + 4.5)).ceil() as i32 + 2;
            for n0 in n0_min..=n0_max {
                let px_mid = 2.5 * f64::from(n0) - 0.5 * hf - wx;
                for n1 in (n1_center.floor() as i32 - n1_span)..=(n1_center.ceil() as i32 + n1_span)
                {
                    let py_mid =
                        (2.5 * f64::from(n1) - 0.5 * hf - px_mid * c1 - wx * c2 - wy * s2) / s1;
                    if px_mid * px_mid + py_mid * py_mid <= (r_norm + 4.5) * (r_norm + 4.5) {
                        self.scan_penrose_fiber(
                            [n0, n1, h],
                            [px_mid, py_mid],
                            p,
                            r_sq,
                            &mut counts,
                        );
                    }
                }
            }
        }
        let thin_f = counts[1].max(1) as f64;
        QuasicrystalStats {
            vertex_count: self.vertices.len(),
            tile_count: self.tiles.len(),
            primary_tile_count: counts[0],
            secondary_tile_count: counts[1],
            tertiary_tile_count: 0,
            flipped_tile_count: counts[3],
            empirical_ratio: counts[0] as f64 / thin_f,
            theoretical_ratio: GOLDEN_RATIO,
            secondary_geometric_ratio: GOLDEN_RATIO,
        }
    }

    #[verified_engine::verified]
    fn scan_penrose_fiber(
        &mut self,
        n01h: [i32; 3],
        p_mid: [f64; 2],
        p: &QuasicrystalParams,
        r_sq: f64,
        counts: &mut [usize; 4],
    ) {
        let [n0, n1, h] = n01h;
        let hf = f64::from(h);
        let [wx, wy] = p.phason;
        let (s1, c1) = (2.0 * PI / 5.0).sin_cos();
        let (s2, c2) = (4.0 * PI / 5.0).sin_cos();
        let n2_mid = 0.2 * hf + 0.4 * (p_mid[0] * c2 + p_mid[1] * s2 + wx * c1 - wy * s1);
        let n3_mid = 0.2 * hf + 0.4 * (p_mid[0] * c2 - p_mid[1] * s2 + wx * c1 + wy * s1);

        for n2 in (n2_mid.floor() as i32 - 3)..=(n2_mid.ceil() as i32 + 3) {
            for n3 in (n3_mid.floor() as i32 - 3)..=(n3_mid.ceil() as i32 + 3) {
                let n = [n0, n1, n2, n3, h - (n0 + n1 + n2 + n3)];
                let Some(perp) = penrose_accept(&n, p.phason) else {
                    continue;
                };
                let par = penrose_parallel(&n, p.edge_length);
                if par[0] * par[0] + par[1] * par[1] <= r_sq {
                    self.push_vertex(par, perp, h as i8);
                }
                if h <= 3 {
                    self.emit_penrose_tiles(&n, p, r_sq, counts);
                }
            }
        }
    }

    #[verified_engine::verified]
    fn emit_penrose_tiles(
        &mut self,
        n: &[i32; 5],
        p: &QuasicrystalParams,
        r_sq: f64,
        counts: &mut [usize; 4],
    ) {
        for i in 0..5 {
            for j in (i + 1)..5 {
                let (mut ni, mut nj) = (*n, *n);
                ni[i] += 1;
                nj[j] += 1;
                let mut nij = ni;
                nij[j] += 1;
                if penrose_accept(&ni, p.phason).is_none()
                    || penrose_accept(&nj, p.phason).is_none()
                    || penrose_accept(&nij, p.phason).is_none()
                {
                    continue;
                }
                let (v0, v3) = (
                    penrose_parallel(n, p.edge_length),
                    penrose_parallel(&nij, p.edge_length),
                );
                if 0.25 * ((v0[0] + v3[0]).powi(2) + (v0[1] + v3[1]).powi(2)) > r_sq {
                    continue;
                }
                let flipped = !(penrose_accept(n, p.reference_phason).is_some()
                    && penrose_accept(&ni, p.reference_phason).is_some()
                    && penrose_accept(&nj, p.reference_phason).is_some()
                    && penrose_accept(&nij, p.reference_phason).is_some());
                let is_thick = (j - i).min(5 - (j - i)) == 1;
                counts[usize::from(!is_thick)] += 1;
                counts[3] += usize::from(flipped);
                self.push_tile(ProjectedTile2D {
                    kind: if is_thick {
                        QuasicrystalTileKind::PenroseThickRhomb
                    } else {
                        QuasicrystalTileKind::PenroseThinRhomb
                    },
                    vertices: [
                        v0,
                        penrose_parallel(&ni, p.edge_length),
                        v3,
                        penrose_parallel(&nj, p.edge_length),
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
    pub(super) fn project_ammann_beenker(&mut self, p: &QuasicrystalParams) -> QuasicrystalStats {
        let a = p.edge_length;
        let r_norm = p.aperture_radius / a;
        let r_sq = p.aperture_radius * p.aperture_radius;
        let [wx, wy] = p.phason;
        let inv_s2 = 1.0 / SQRT_2;
        let n0_min = (0.5 * (-r_norm - 3.5 + wx)).floor() as i32 - 1;
        let n0_max = (0.5 * (r_norm + 3.5 + wx)).ceil() as i32 + 1;
        let n2_min = (0.5 * (-r_norm - 3.5 - wy)).floor() as i32 - 1;
        let n2_max = (0.5 * (r_norm + 3.5 - wy)).ceil() as i32 + 1;
        let mut counts = [0_usize; 4];

        for n0 in n0_min..=n0_max {
            let px_mid = 2.0 * f64::from(n0) - wx;
            for n2 in n2_min..=n2_max {
                let py_mid = 2.0 * f64::from(n2) + wy;
                if px_mid * px_mid + py_mid * py_mid > (r_norm + 3.5) * (r_norm + 3.5) {
                    continue;
                }
                let n1_mid = 0.5 * inv_s2 * (px_mid + py_mid - wx + wy);
                let n3_mid = 0.5 * inv_s2 * (-px_mid + py_mid + wx + wy);
                for n1 in (n1_mid.floor() as i32 - 2)..=(n1_mid.ceil() as i32 + 2) {
                    for n3 in (n3_mid.floor() as i32 - 2)..=(n3_mid.ceil() as i32 + 2) {
                        let n = [n0, n1, n2, n3];
                        if let Some(perp) = ammann_accept(&n, p.phason) {
                            let par = ammann_parallel(&n, a);
                            if par[0] * par[0] + par[1] * par[1] <= r_sq {
                                self.push_vertex(par, perp, 0);
                            }
                            self.emit_ammann_tiles(&n, p, r_sq, &mut counts);
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
            tertiary_tile_count: 0,
            flipped_tile_count: counts[3],
            empirical_ratio: (counts[0] + counts[1]) as f64 / sq_f,
            theoretical_ratio: SILVER_RATIO,
            secondary_geometric_ratio: SILVER_RATIO,
        }
    }

    #[verified_engine::verified]
    fn emit_ammann_tiles(
        &mut self,
        n: &[i32; 4],
        p: &QuasicrystalParams,
        r_sq: f64,
        counts: &mut [usize; 4],
    ) {
        for i in 0..4 {
            for j in (i + 1)..4 {
                let (mut ni, mut nj) = (*n, *n);
                ni[i] += 1;
                nj[j] += 1;
                let mut nij = ni;
                nij[j] += 1;
                if ammann_accept(&ni, p.phason).is_none()
                    || ammann_accept(&nj, p.phason).is_none()
                    || ammann_accept(&nij, p.phason).is_none()
                {
                    continue;
                }
                let (v0, v3) = (
                    ammann_parallel(n, p.edge_length),
                    ammann_parallel(&nij, p.edge_length),
                );
                if 0.25 * ((v0[0] + v3[0]).powi(2) + (v0[1] + v3[1]).powi(2)) > r_sq {
                    continue;
                }
                let flipped = !(ammann_accept(n, p.reference_phason).is_some()
                    && ammann_accept(&ni, p.reference_phason).is_some()
                    && ammann_accept(&nj, p.reference_phason).is_some()
                    && ammann_accept(&nij, p.reference_phason).is_some());
                let is_sq = j - i == 2;
                counts[usize::from(is_sq)] += 1;
                counts[3] += usize::from(flipped);
                self.push_tile(ProjectedTile2D {
                    kind: if is_sq {
                        QuasicrystalTileKind::AmmannSquare
                    } else {
                        QuasicrystalTileKind::AmmannRhomb
                    },
                    vertices: [
                        v0,
                        ammann_parallel(&ni, p.edge_length),
                        v3,
                        ammann_parallel(&nj, p.edge_length),
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                    vertex_count: 4,
                    phason_flipped: flipped,
                });
            }
        }
    }
}

#[verified_engine::verified]
fn penrose_parallel(n: &[i32; 5], a: f64) -> [f64; 2] {
    let (mut x, mut y) = (0.0, 0.0);
    for (j, &nj) in n.iter().enumerate() {
        let (s, c) = (2.0 * PI * (j as f64) / 5.0).sin_cos();
        x += f64::from(nj) * c;
        y += f64::from(nj) * s;
    }
    [a * x, a * y]
}

#[verified_engine::verified]
fn penrose_slice_bounds(z: f64) -> (f64, f64) {
    let c72 = 0.5 * (GOLDEN_RATIO - 1.0);
    let c144 = -0.5 * GOLDEN_RATIO;
    let hi_slopes = [1.0, c72, c72, c144, c144];
    let lo_slopes = [c144, c144, c72, c72, 1.0];
    let (mut rem_hi, mut rem_lo, mut hi, mut lo) = (z, z, 0.0, 0.0);
    for idx in 0..5 {
        let t_hi = rem_hi.clamp(0.0, 1.0);
        let t_lo = rem_lo.clamp(0.0, 1.0);
        hi += t_hi * hi_slopes[idx];
        lo += t_lo * lo_slopes[idx];
        rem_hi -= t_hi;
        rem_lo -= t_lo;
    }
    (lo, hi)
}

#[verified_engine::verified]
fn penrose_accept(n: &[i32; 5], phason: [f64; 2]) -> Option<[f64; 2]> {
    let h: i32 = n.iter().sum();
    let z = f64::from(h) - PENROSE_GAMMA0;
    if !(0.0..5.0).contains(&z) {
        return None;
    }
    let (mut yx, mut yy) = (
        -(phason[0] + NON_SINGULAR_BIAS[0]),
        -(phason[1] + NON_SINGULAR_BIAS[1]),
    );
    for (j, &nj) in n.iter().enumerate() {
        let (s, c) = (4.0 * PI * (j as f64) / 5.0).sin_cos();
        yx += f64::from(nj) * c;
        yy += f64::from(nj) * s;
    }
    let (lo, hi) = penrose_slice_bounds(z);
    for k in 0..5 {
        let (s, c) = (2.0 * PI * (k as f64) / 5.0).sin_cos();
        let proj = yx * c + yy * s;
        if proj < lo || proj > hi {
            return None;
        }
    }
    Some([yx, yy])
}

#[verified_engine::verified]
fn ammann_parallel(n: &[i32; 4], a: f64) -> [f64; 2] {
    let s = 1.0 / SQRT_2;
    let nf = [
        f64::from(n[0]),
        f64::from(n[1]),
        f64::from(n[2]),
        f64::from(n[3]),
    ];
    [
        a * (nf[0] + s * (nf[1] - nf[3])),
        a * (nf[2] + s * (nf[1] + nf[3])),
    ]
}

#[verified_engine::verified]
fn ammann_accept(n: &[i32; 4], phason: [f64; 2]) -> Option<[f64; 2]> {
    let s = 1.0 / SQRT_2;
    let nf = [
        f64::from(n[0]),
        f64::from(n[1]),
        f64::from(n[2]),
        f64::from(n[3]),
    ];
    let yx = nf[0] + s * (-nf[1] + nf[3]) - (phason[0] + NON_SINGULAR_BIAS[0]);
    let yy = -nf[2] + s * (nf[1] + nf[3]) - (phason[1] + NON_SINGULAR_BIAS[1]);
    let bound = 0.5 * SILVER_RATIO;
    for k in 0..4 {
        let (sn, cs) = ((k as f64) * FRAC_PI_4).sin_cos();
        if (yx * cs + yy * sn).abs() > bound {
            return None;
        }
    }
    Some([yx, yy])
}
