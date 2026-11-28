//! Planar projection from [`ClosedLink3D`], 3D diagram lifting, and canonical presets.

// [cite:knot_braid_topology]

use super::KnotError;
use super::curves::ClosedLink3D;
use super::diagram::PlanarDiagram;
use crate::pure_math::algebra::BraidWord;

mod lift_spline;
mod lift_tiers;
mod project_3d;

impl PlanarDiagram {
    /// Canonical 3-crossing right-handed (`positive == true`) or left-handed trefoil diagram.
    #[must_use]
    #[verified_engine::verified]
    pub fn trefoil(positive: bool) -> Self {
        let g = if positive { 1 } else { -1 };
        let braid = BraidWord::new(2, vec![g, g, g]).expect("valid 2-strand trefoil braid");
        Self::from_braid(&braid)
    }

    /// Canonical 4-crossing alternating figure-eight ($4_1$) diagram.
    #[must_use]
    #[verified_engine::verified]
    pub fn figure_eight() -> Self {
        let braid = BraidWord::new(3, vec![1, -2, 1, -2]).expect("valid 3-strand figure-8 braid");
        Self::from_braid(&braid)
    }

    /// Canonical 2-crossing Hopf link diagram (`positive` controls linking number $\pm 1$).
    #[must_use]
    #[verified_engine::verified]
    pub fn hopf_link(positive: bool) -> Self {
        let g = if positive { 1 } else { -1 };
        let braid = BraidWord::new(2, vec![g, g]).expect("valid 2-strand Hopf braid");
        Self::from_braid(&braid)
    }

    /// Canonical 6-crossing alternating Borromean rings ($6^3_2$) diagram.
    #[must_use]
    #[verified_engine::verified]
    pub fn borromean_rings() -> Self {
        let braid =
            BraidWord::new(3, vec![1, -2, 1, -2, 1, -2]).expect("valid 3-strand Borromean braid");
        Self::from_braid(&braid)
    }

    /// Projects a generic 3D [`ClosedLink3D`] onto the $(x, y)$ plane to extract an oriented [`PlanarDiagram`].
    ///
    /// Applies deterministic orthogonal micro-tilts $R \in SO(3)$ when polyline vertices touch
    /// crossing rays or parallel segment projections overlap.
    ///
    /// # Errors
    /// Returns [`KnotError`] if the projection exceeds [`super::diagram::MAX_DIAGRAM_CROSSINGS`]
    /// or exhibits unresolvable degenerate crossings.
    #[verified_engine::verified]
    pub fn from_3d_link(link: &ClosedLink3D) -> Result<Self, KnotError> {
        project_3d::project_3d_link_to_diagram(link)
    }

    /// Lifts `self` into a smooth 3D [`ClosedLink3D`] via periodic centripetal Catmull-Rom
    /// splines interpolating exact `[x, y, +-z_bridge]` crossing bridges and routing any
    /// non-crossing 2D overlaps onto disjoint $z$ tiers.
    ///
    /// # Errors
    /// Returns [`KnotError`] if `samples_per_segment < 4` or `self.validate()` fails.
    #[verified_engine::verified]
    pub fn lift_to_3d_link(&self, samples_per_segment: usize) -> Result<ClosedLink3D, KnotError> {
        self.validate()?;
        if samples_per_segment < 4 {
            return Err(KnotError::InsufficientSamples {
                required: 4,
                actual: samples_per_segment,
            });
        }
        lift_spline::lift_diagram_to_3d(self, samples_per_segment)
    }
}
