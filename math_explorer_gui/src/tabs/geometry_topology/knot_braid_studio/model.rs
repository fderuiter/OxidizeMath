//! Computed 3D link geometry, Bishop tubular polylines, planar diagrams, and Laurent invariants.

// [cite:knot_braid_topology]

use super::{KnotBraidStudioTool, MAX_REIDEMEISTER_HISTORY};
use math_explorer::pure_math::algebra::{BraidWord, LaurentPolynomial};
use math_explorer::pure_math::knot_theory::{
    alexander_polynomial_burau, alexander_polynomial_fox, borromean_rings, compute_bishop_frames,
    figure_eight_knot, format_jones_polynomial, gauss_writhe_quadrature, hopf_link,
    is_chiral_by_jones, jones_polynomial_half_power, kauffman_bracket, pairwise_linking_matrix,
    BishopFrame, ClosedLink3D, CrossingSign, KnotError, LissajousKnot, PlanarDiagram,
    ReidemeisterTriangle, TorusKnot,
};
use nalgebra::Point3;
use std::f64::consts::TAU;

/// Selectable 3D knot, link, or Artin braid family in the studio.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum StudioFamily {
    ArtinBraid,
    TorusKnot,
    LissajousKnot,
    FigureEight,
    HopfLink,
    BorromeanRings,
}

impl StudioFamily {
    pub const ALL: [Self; 6] = [
        Self::ArtinBraid,
        Self::TorusKnot,
        Self::LissajousKnot,
        Self::FigureEight,
        Self::HopfLink,
        Self::BorromeanRings,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::ArtinBraid => "Artin Braid Closure",
            Self::TorusKnot => "Torus Knot T(p,q)",
            Self::LissajousKnot => "Lissajous Knot",
            Self::FigureEight => "Figure-Eight (4_1)",
            Self::HopfLink => "Hopf Link (2^2_1)",
            Self::BorromeanRings => "Borromean Rings (6^3_2)",
        }
    }
}

/// Compact serializable Reidemeister ambient isotopy step replayed onto the active diagram.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(tag = "op")]
pub enum DiagramEditOp {
    RiAdd {
        segment: usize,
        positive: bool,
    },
    RiRemove {
        crossing: usize,
    },
    RiiAdd {
        over_seg: usize,
        under_seg: usize,
    },
    RiiRemove {
        c1: usize,
        c2: usize,
    },
    RiiiSlide {
        c_tb: usize,
        c_tm: usize,
        c_mb: usize,
    },
}

pub(crate) type ComponentPolyline = (usize, Vec<Point3<f64>>);

/// Cached geometry and exact topological invariants for the active studio configuration.
pub struct StudioComputedModel {
    pub link_3d: ClosedLink3D,
    pub bishop_frames: Vec<Vec<BishopFrame>>,
    pub tube_rails: Vec<ComponentPolyline>,
    pub tube_rings: Vec<ComponentPolyline>,
    pub diagram: PlanarDiagram,
    pub riii_triangles: Vec<ReidemeisterTriangle>,
    pub writhe_diagram: i64,
    pub writhe_gauss_3d: Vec<f64>,
    pub linking_matrix: Vec<Vec<i64>>,
    pub alexander_fox: Option<LaurentPolynomial<i64>>,
    pub alexander_burau: Option<LaurentPolynomial<i64>>,
    pub kauffman_bracket: Option<LaurentPolynomial<i64>>,
    pub jones_half_power: Option<LaurentPolynomial<i64>>,
    pub jones_display: String,
    pub is_chiral: Option<bool>,
    pub crossing_cap_notice: Option<String>,
}

impl StudioComputedModel {
    pub fn empty() -> Self {
        let link_3d = figure_eight_knot(2.0, 24).expect("default fallback link");
        Self {
            link_3d,
            bishop_frames: Vec::new(),
            tube_rails: Vec::new(),
            tube_rings: Vec::new(),
            diagram: PlanarDiagram::unknot(),
            riii_triangles: Vec::new(),
            writhe_diagram: 0,
            writhe_gauss_3d: Vec::new(),
            linking_matrix: Vec::new(),
            alexander_fox: None,
            alexander_burau: None,
            kauffman_bracket: None,
            jones_half_power: None,
            jones_display: String::new(),
            is_chiral: None,
            crossing_cap_notice: None,
        }
    }
}

impl KnotBraidStudioTool {
    pub(crate) fn geometry_hash(&self) -> u64 {
        use std::hash::{Hash as _, Hasher as _};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.tube_radius.to_bits().hash(&mut h);
        self.family.hash(&mut h);
        self.braid_strands.hash(&mut h);
        self.braid_generators.hash(&mut h);
        self.torus_p.hash(&mut h);
        self.torus_q.hash(&mut h);
        self.lissajous_freqs.hash(&mut h);
        for p in self.lissajous_phases {
            p.to_bits().hash(&mut h);
        }
        self.major_radius.to_bits().hash(&mut h);
        self.minor_radius.to_bits().hash(&mut h);
        self.curve_samples.hash(&mut h);
        self.radial_segments.hash(&mut h);
        self.reidemeister_ops.hash(&mut h);
        h.finish().max(1)
    }

    pub(crate) fn sync_model(&mut self) {
        let key = self.geometry_hash();
        if self.cached_key == key {
            return;
        }
        self.cached_key = key;
        self.model = self.build_computed_model();
        let c_len = self.model.diagram.crossings.len();
        if c_len > 0 {
            self.selected_crossing %= c_len;
            self.second_crossing %= c_len;
        } else {
            self.selected_crossing = 0;
            self.second_crossing = 0;
        }
        let s_len = self.model.diagram.segments.len();
        if s_len > 0 {
            self.selected_segment %= s_len;
            self.second_segment %= s_len;
        } else {
            self.selected_segment = 0;
            self.second_segment = 0;
        }
        if !self.braid_generators.is_empty() {
            self.selected_braid_idx %= self.braid_generators.len();
        } else {
            self.selected_braid_idx = 0;
        }
    }

    fn build_base_link_and_diagram(&self) -> (ClosedLink3D, PlanarDiagram, Option<BraidWord>) {
        match self.family {
            StudioFamily::ArtinBraid => {
                let braid = BraidWord::new(self.braid_strands, self.braid_generators.clone())
                    .unwrap_or_else(|_| BraidWord::identity(2).expect("2-strand identity"));
                let link = ClosedLink3D::from_braid_closure(
                    &braid,
                    self.major_radius,
                    0.34,
                    self.curve_samples.max(24),
                )
                .unwrap_or_else(|_| figure_eight_knot(2.0, 48).expect("fallback link"));
                let diag = PlanarDiagram::from_braid(&braid);
                (link, diag, Some(braid))
            }
            StudioFamily::TorusKnot => self.build_torus_family(),
            StudioFamily::LissajousKnot => self.build_lissajous_family(),
            StudioFamily::FigureEight => {
                let braid = BraidWord::new(3, vec![1, -2, 1, -2]).expect("figure-eight braid");
                let link = figure_eight_knot(self.major_radius, self.curve_samples.max(48))
                    .expect("figure-eight 3d");
                (link, PlanarDiagram::figure_eight(), Some(braid))
            }
            StudioFamily::HopfLink => {
                let braid = BraidWord::new(2, vec![1, 1]).expect("hopf braid");
                let link = hopf_link(self.major_radius * 0.65, true, self.curve_samples.max(40))
                    .expect("hopf 3d");
                (link, PlanarDiagram::hopf_link(true), Some(braid))
            }
            StudioFamily::BorromeanRings => {
                let braid = BraidWord::new(3, vec![1, -2, 1, -2, 1, -2]).expect("borromean braid");
                let link = borromean_rings(
                    self.major_radius,
                    (self.major_radius * 0.52).max(0.45),
                    self.curve_samples.max(56),
                )
                .expect("borromean 3d");
                (link, PlanarDiagram::borromean_rings(), Some(braid))
            }
        }
    }

    fn build_torus_family(&self) -> (ClosedLink3D, PlanarDiagram, Option<BraidWord>) {
        let minor = self.minor_radius.min(self.major_radius * 0.48);
        let q_nz = if self.torus_q == 0 { 1 } else { self.torus_q };
        let torus = TorusKnot::new(self.torus_p, q_nz, self.major_radius, minor)
            .unwrap_or_else(|_| TorusKnot::right_trefoil());
        let link = torus
            .to_link(self.curve_samples.max(48))
            .unwrap_or_else(|_| TorusKnot::right_trefoil().to_link(64).expect("trefoil"));
        let braid_opt = build_torus_braid(self.torus_p, q_nz);
        let diag = braid_opt
            .as_ref()
            .map(PlanarDiagram::from_braid)
            .or_else(|| PlanarDiagram::from_3d_link(&link).ok())
            .unwrap_or_else(|| PlanarDiagram::trefoil(q_nz > 0));
        (link, diag, braid_opt)
    }

    fn build_lissajous_family(&self) -> (ClosedLink3D, PlanarDiagram, Option<BraidWord>) {
        let [fx, fy, fz] = self.lissajous_freqs;
        let [px, py, pz] = self.lissajous_phases;
        let liss = LissajousKnot::new((fx, fy, fz), (px, py, pz), self.major_radius * 0.9)
            .unwrap_or_else(|_| LissajousKnot::three_twist());
        let link = liss
            .to_link(self.curve_samples.max(64))
            .unwrap_or_else(|_| figure_eight_knot(2.0, 64).expect("fallback"));
        let diag = PlanarDiagram::from_3d_link(&link).unwrap_or_else(|_| PlanarDiagram::unknot());
        (link, diag, None)
    }

    fn build_computed_model(&self) -> StudioComputedModel {
        let (mut link_3d, mut diagram, active_braid) = self.build_base_link_and_diagram();
        if !self.reidemeister_ops.is_empty() {
            for &op in &self.reidemeister_ops {
                let _ = apply_edit_op(&mut diagram, op);
            }
            if let Ok(lifted) = diagram.lift_to_3d_link(10) {
                link_3d = lifted;
            }
        }
        let mut bishop_frames = Vec::with_capacity(link_3d.components.len());
        for comp in &link_3d.components {
            bishop_frames.push(compute_bishop_frames(comp).unwrap_or_default());
        }
        let (tube_rails, tube_rings) =
            build_tube_wireframe(&link_3d, &bishop_frames, self.tube_radius);
        let writhe_gauss_3d = link_3d
            .components
            .iter()
            .map(|c| gauss_writhe_quadrature(c).unwrap_or(0.0))
            .collect();
        let linking_matrix = pairwise_linking_matrix(&link_3d)
            .unwrap_or_else(|_| diagram.combinatorial_linking_matrix());
        populate_invariants(
            TubeGeometryBundle {
                link_3d,
                bishop_frames,
                tube_rails,
                tube_rings,
            },
            diagram,
            active_braid,
            writhe_gauss_3d,
            linking_matrix,
        )
    }

    pub(crate) fn push_reidemeister_op(&mut self, op: DiagramEditOp) -> Result<(), String> {
        if self.reidemeister_ops.len() >= MAX_REIDEMEISTER_HISTORY {
            return Err(format!(
                "Reidemeister history capped at {MAX_REIDEMEISTER_HISTORY} moves for 4 KiB URL sharing"
            ));
        }
        let mut probe = self.model.diagram.clone();
        apply_edit_op(&mut probe, op).map_err(|e| e.to_string())?;
        self.reidemeister_ops.push(op);
        self.cached_key = 0;
        self.sync_model();
        Ok(())
    }
}

fn apply_edit_op(diagram: &mut PlanarDiagram, op: DiagramEditOp) -> Result<(), KnotError> {
    match op {
        DiagramEditOp::RiAdd { segment, positive } => {
            let sign = if positive {
                CrossingSign::Positive
            } else {
                CrossingSign::Negative
            };
            diagram.apply_reidemeister_i_add(segment, sign).map(|_| ())
        }
        DiagramEditOp::RiRemove { crossing } => diagram.apply_reidemeister_i_remove(crossing),
        DiagramEditOp::RiiAdd {
            over_seg,
            under_seg,
        } => diagram
            .apply_reidemeister_ii_add(over_seg, under_seg)
            .map(|_| ()),
        DiagramEditOp::RiiRemove { c1, c2 } => diagram.apply_reidemeister_ii_remove(c1, c2),
        DiagramEditOp::RiiiSlide { c_tb, c_tm, c_mb } => {
            diagram.apply_reidemeister_iii(c_tb, c_tm, c_mb)
        }
    }
}

fn build_torus_braid(p: usize, q: i32) -> Option<BraidWord> {
    let abs_q = q.unsigned_abs() as usize;
    let sign = q.signum();
    let total = p.saturating_sub(1).saturating_mul(abs_q);
    if !(2..=8).contains(&p) || total == 0 || total > 64 {
        return None;
    }
    let mut gens = Vec::with_capacity(total);
    for _ in 0..abs_q {
        for i in 1..p {
            gens.push((i as i32) * sign);
        }
    }
    BraidWord::new(p, gens).ok()
}

fn build_tube_wireframe(
    link: &ClosedLink3D,
    frames: &[Vec<BishopFrame>],
    radius: f64,
) -> (Vec<ComponentPolyline>, Vec<ComponentPolyline>) {
    let mut rails = Vec::new();
    let mut rings = Vec::new();
    for (comp_idx, (pts, comp_frames)) in link.components.iter().zip(frames.iter()).enumerate() {
        if pts.len() != comp_frames.len() || pts.is_empty() {
            continue;
        }
        for rail_k in 0..4 {
            let phi = TAU * (rail_k as f64) / 4.0;
            let (sp, cp) = phi.sin_cos();
            let mut line = Vec::with_capacity(pts.len() + 1);
            for (p, f) in pts.iter().zip(comp_frames.iter()) {
                line.push(p + (f.normal_u * cp + f.normal_v * sp) * radius);
            }
            line.push(line[0]);
            rails.push((comp_idx, line));
        }
        let step = (pts.len() / 12).max(3);
        for (i, (p, f)) in pts.iter().zip(comp_frames.iter()).enumerate().step_by(step) {
            let _ = i;
            let mut ring = Vec::with_capacity(9);
            for k in 0..=8 {
                let phi = TAU * (k as f64) / 8.0;
                let (sp, cp) = phi.sin_cos();
                ring.push(p + (f.normal_u * cp + f.normal_v * sp) * radius);
            }
            rings.push((comp_idx, ring));
        }
    }
    (rails, rings)
}

struct TubeGeometryBundle {
    link_3d: ClosedLink3D,
    bishop_frames: Vec<Vec<BishopFrame>>,
    tube_rails: Vec<ComponentPolyline>,
    tube_rings: Vec<ComponentPolyline>,
}

fn populate_invariants(
    geom: TubeGeometryBundle,
    diagram: PlanarDiagram,
    active_braid: Option<BraidWord>,
    writhe_gauss_3d: Vec<f64>,
    linking_matrix: Vec<Vec<i64>>,
) -> StudioComputedModel {
    let riii_triangles = diagram.find_reidemeister_iii_triangles();
    let writhe_diagram = diagram.writhe();
    let alexander_fox = alexander_polynomial_fox(&diagram).ok();
    let alexander_burau = active_braid
        .as_ref()
        .and_then(|b| alexander_polynomial_burau(b).ok());

    let (kauffman_bracket, jones_half_power, jones_display, is_chiral, crossing_cap_notice) =
        match kauffman_bracket(&diagram) {
            Ok(kb) => {
                let jh = jones_polynomial_half_power(&diagram).ok();
                let j_str = format_jones_polynomial(&diagram).unwrap_or_else(|_| "n/a".to_string());
                let chiral = is_chiral_by_jones(&diagram).ok();
                (Some(kb), jh, j_str, chiral, None)
            }
            Err(KnotError::CrossingCapExceeded {
                crossings,
                max_crossings,
            }) => (
                None,
                None,
                format!("capped (>{max_crossings} crossings)"),
                None,
                Some(format!(
                    "Diagram has {crossings} crossings (exceeding exact state-sum cap of \
                     {max_crossings}). Polynomial-time Gauss writhe, linking numbers, and \
                     Fox/Burau Alexander polynomials remain active."
                )),
            ),
            Err(err) => (
                None,
                None,
                "unavailable".to_string(),
                None,
                Some(err.to_string()),
            ),
        };

    StudioComputedModel {
        link_3d: geom.link_3d,
        bishop_frames: geom.bishop_frames,
        tube_rails: geom.tube_rails,
        tube_rings: geom.tube_rings,
        diagram,
        riii_triangles,
        writhe_diagram,
        writhe_gauss_3d,
        linking_matrix,
        alexander_fox,
        alexander_burau,
        kauffman_bracket,
        jones_half_power,
        jones_display,
        is_chiral,
        crossing_cap_notice,
    }
}
