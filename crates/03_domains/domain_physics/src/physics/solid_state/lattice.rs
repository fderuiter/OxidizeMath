use nalgebra::Vector3;

/// Strategy trait for generating crystal unit cells.
pub trait CrystalSystem {
    /// Generates the unit cell for the given lattice constant.
    #[verified_engine::verified]
    fn generate(&self, a: f64) -> UnitCell;
}

/// Simple Cubic (SC) crystal system.
#[derive(Debug, Clone, Copy, Default)]
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
#[derive(Debug, Clone, Copy, Default)]
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
#[derive(Debug, Clone, Copy, Default)]
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
#[derive(Debug, Clone)]
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
