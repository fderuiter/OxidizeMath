use super::{PersistenceInterval, Point2D, PointCloud, Simplex, SimplicialComplex};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

impl Serialize for PointCloud {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct PointCloudRef<'a> {
            points: &'a [Point2D],
        }
        PointCloudRef {
            points: &self.points,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PointCloud {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct PointCloudRaw {
            points: Vec<Point2D>,
        }
        let raw = PointCloudRaw::deserialize(deserializer)?;
        PointCloud::new(raw.points).map_err(serde::de::Error::custom)
    }
}

impl Serialize for Simplex {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct SimplexRef<'a> {
            vertices: &'a [usize],
        }
        SimplexRef {
            vertices: &self.vertices,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Simplex {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct SimplexRaw {
            vertices: Vec<usize>,
        }
        let raw = SimplexRaw::deserialize(deserializer)?;
        Simplex::new(raw.vertices).map_err(serde::de::Error::custom)
    }
}

impl Serialize for SimplicialComplex {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct ComplexRef<'a> {
            simplices: &'a [std::collections::HashSet<Simplex>],
        }
        ComplexRef {
            simplices: &self.simplices,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SimplicialComplex {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ComplexRaw {
            simplices: Vec<Vec<Simplex>>,
        }
        let raw = ComplexRaw::deserialize(deserializer)?;
        let mut complex = SimplicialComplex::new();
        for level in raw.simplices {
            for simplex in level {
                complex
                    .add_simplex(simplex)
                    .map_err(serde::de::Error::custom)?;
            }
        }
        Ok(complex)
    }
}

impl Serialize for PersistenceInterval {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct IntervalRaw {
            birth: f64,
            death: f64,
            dimension: usize,
        }
        IntervalRaw {
            birth: self.birth,
            death: self.death,
            dimension: self.dimension,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PersistenceInterval {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct IntervalRaw {
            birth: f64,
            death: f64,
            dimension: usize,
        }
        let raw = IntervalRaw::deserialize(deserializer)?;
        PersistenceInterval::new(raw.birth, raw.death, raw.dimension)
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        PersistenceBarcode, PersistenceInterval, Point2D, PointCloud, Simplex, SimplicialComplex,
    };

    #[test]
    #[verified_engine::verified]
    fn test_tda_serde_validation() {
        let interval = PersistenceInterval::new(0.2, 1.5, 1).unwrap();
        let mut barcode = PersistenceBarcode::new();
        barcode.add_interval(interval);
        let json = serde_json::to_string(&barcode).unwrap();
        let restored: PersistenceBarcode = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, barcode);

        assert!(
            serde_json::from_str::<PersistenceInterval>(
                r#"{"birth":2.0,"death":1.0,"dimension":0}"#
            )
            .is_err()
        );
        assert!(serde_json::from_str::<PointCloud>(r#"{"points":[]}"#).is_err());
        assert!(serde_json::from_str::<Simplex>(r#"{"vertices":[1,1]}"#).is_err());

        let s: Simplex = serde_json::from_str(r#"{"vertices":[2,0,1]}"#).unwrap();
        assert_eq!(s.vertices, vec![0, 1, 2]);

        let mut complex = SimplicialComplex::new();
        complex.add_simplex(s).unwrap();
        let complex_json = serde_json::to_string(&complex).unwrap();
        let complex_restored: SimplicialComplex = serde_json::from_str(&complex_json).unwrap();
        assert_eq!(complex_restored.count_simplices(0), 3);

        let cloud = PointCloud::new(vec![Point2D::new(0.0, 1.0)]).unwrap();
        let cloud_json = serde_json::to_string(&cloud).unwrap();
        let cloud_restored: PointCloud = serde_json::from_str(&cloud_json).unwrap();
        assert_eq!(cloud_restored.size(), 1);
    }
}
