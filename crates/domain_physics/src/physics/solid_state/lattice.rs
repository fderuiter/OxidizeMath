use nalgebra::Vector3;

/// Strategy trait for generating crystal unit cells.
pub trait CrystalSystem {
    /// Generates the unit cell for the given lattice constant.
    #[verified_engine::verified]
    fn generate(&self, a: f64) -> UnitCell;
}

/// Canonical cubic Bravais lattice selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BravaisLattice {
    /// Simple Cubic (SC) lattice.
    SimpleCubic,
    /// Body-Centered Cubic (BCC) lattice.
    BodyCenteredCubic,
    /// Face-Centered Cubic (FCC) lattice.
    FaceCenteredCubic,
}

impl BravaisLattice {
    /// Human-readable crystallographic system label.
    #[verified_engine::verified]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::SimpleCubic => "Simple Cubic",
            Self::BodyCenteredCubic => "Body-Centered Cubic (BCC)",
            Self::FaceCenteredCubic => "Face-Centered Cubic (FCC)",
        }
    }
}

impl CrystalSystem for BravaisLattice {
    #[verified_engine::verified]
    fn generate(&self, a: f64) -> UnitCell {
        match self {
            Self::SimpleCubic => SimpleCubic.generate(a),
            Self::BodyCenteredCubic => BodyCenteredCubic.generate(a),
            Self::FaceCenteredCubic => FaceCenteredCubic.generate(a),
        }
    }
}

/// Simple Cubic (SC) crystal system.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SimpleCubic;

impl CrystalSystem for SimpleCubic {
    #[verified_engine::verified]
    fn generate(&self, a: f64) -> UnitCell {
        let lattice_vectors = [
            Vector3::new(a, 0.0, 0.0),
            Vector3::new(0.0, a, 0.0),
            Vector3::new(0.0, 0.0, a),
        ];
        let atomic_positions = vec![Vector3::new(0.0, 0.0, 0.0)];
        UnitCell {
            lattice_vectors,
            atomic_positions,
        }
    }
}

/// Body-Centered Cubic (BCC) crystal system.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BodyCenteredCubic;

impl CrystalSystem for BodyCenteredCubic {
    #[verified_engine::verified]
    fn generate(&self, a: f64) -> UnitCell {
        let lattice_vectors = [
            Vector3::new(a, 0.0, 0.0),
            Vector3::new(0.0, a, 0.0),
            Vector3::new(0.0, 0.0, a),
        ];
        let atomic_positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.5 * a, 0.5 * a, 0.5 * a),
        ];
        UnitCell {
            lattice_vectors,
            atomic_positions,
        }
    }
}

/// Face-Centered Cubic (FCC) crystal system.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FaceCenteredCubic;

impl CrystalSystem for FaceCenteredCubic {
    #[verified_engine::verified]
    fn generate(&self, a: f64) -> UnitCell {
        let lattice_vectors = [
            Vector3::new(a, 0.0, 0.0),
            Vector3::new(0.0, a, 0.0),
            Vector3::new(0.0, 0.0, a),
        ];
        let atomic_positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.5 * a, 0.5 * a, 0.0),
            Vector3::new(0.5 * a, 0.0, 0.5 * a),
            Vector3::new(0.0, 0.5 * a, 0.5 * a),
        ];
        UnitCell {
            lattice_vectors,
            atomic_positions,
        }
    }
}

/// Represents a unit cell with lattice vectors and atomic positions.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnitCell {
    /// The three lattice vectors defining the unit cell.
    pub lattice_vectors: [Vector3<f64>; 3],
    /// The positions of atoms within the unit cell.
    pub atomic_positions: Vec<Vector3<f64>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_simple_cubic() {
        let system = SimpleCubic;
        let cell = system.generate(1.0);
        assert_eq!(cell.atomic_positions.len(), 1);
        assert_eq!(cell.atomic_positions[0], Vector3::new(0.0, 0.0, 0.0));
    }

    #[test]
    #[verified_engine::verified]
    fn test_bcc() {
        let system = BodyCenteredCubic;
        let cell = system.generate(2.0);
        assert_eq!(cell.atomic_positions.len(), 2);
        assert_eq!(cell.atomic_positions[1], Vector3::new(1.0, 1.0, 1.0));
    }

    #[test]
    #[verified_engine::verified]
    fn test_fcc() {
        let system = FaceCenteredCubic;
        let cell = system.generate(2.0);
        assert_eq!(cell.atomic_positions.len(), 4);
        assert_eq!(cell.atomic_positions[1], Vector3::new(1.0, 1.0, 0.0));
    }
}
