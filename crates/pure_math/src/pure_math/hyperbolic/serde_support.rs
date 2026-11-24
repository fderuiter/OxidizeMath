// [cite:pure_math]
use super::{BilliardDomainKind, BilliardTable, DiskPoint, HalfPlanePoint, Psl2R, Psu11};
use num_complex::Complex64;

impl serde::Serialize for BilliardTable {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(serde::Serialize)]
        struct RawBilliardTable {
            kind: BilliardDomainKind,
        }

        RawBilliardTable { kind: self.kind() }.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for BilliardTable {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RawBilliardTable {
            kind: BilliardDomainKind,
        }

        let raw = RawBilliardTable::deserialize(deserializer)?;
        Self::new(raw.kind).map_err(serde::de::Error::custom)
    }
}

impl<'de> serde::Deserialize<'de> for DiskPoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RawDiskPoint {
            re: f64,
            im: f64,
        }

        let raw = RawDiskPoint::deserialize(deserializer)?;
        Self::new(raw.re, raw.im).map_err(serde::de::Error::custom)
    }
}

impl<'de> serde::Deserialize<'de> for HalfPlanePoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RawHalfPlanePoint {
            re: f64,
            im: f64,
        }

        let raw = RawHalfPlanePoint::deserialize(deserializer)?;
        Self::new(raw.re, raw.im).map_err(serde::de::Error::custom)
    }
}

impl<'de> serde::Deserialize<'de> for Psu11 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RawPsu11 {
            a_re: f64,
            a_im: f64,
            b_re: f64,
            b_im: f64,
        }

        let raw = RawPsu11::deserialize(deserializer)?;
        Self::new(
            Complex64::new(raw.a_re, raw.a_im),
            Complex64::new(raw.b_re, raw.b_im),
        )
        .map_err(serde::de::Error::custom)
    }
}

impl<'de> serde::Deserialize<'de> for Psl2R {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RawPsl2R {
            alpha: f64,
            beta: f64,
            gamma: f64,
            delta: f64,
        }

        let raw = RawPsl2R::deserialize(deserializer)?;
        Self::new(raw.alpha, raw.beta, raw.gamma, raw.delta).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn disk_and_half_plane_points_enforce_domain_invariants_on_deserialize() {
        let p = DiskPoint::new(0.25, -0.5).expect("valid disk point");
        let p_json = serde_json::to_string(&p).expect("serialize disk point");
        let p_back: DiskPoint = serde_json::from_str(&p_json).expect("deserialize disk point");
        assert_eq!(p_back, p);
        assert!(serde_json::from_str::<DiskPoint>(r#"{"re":1.0,"im":0.0}"#).is_err());
        assert!(serde_json::from_str::<DiskPoint>(r#"{"re":0.8,"im":0.8}"#).is_err());
        assert!(serde_json::from_str::<DiskPoint>(r#"{"re":0.0,"im":0.0,"extra":1}"#).is_err());

        let q = HalfPlanePoint::new(-1.25, 2.5).expect("valid half-plane point");
        let q_json = serde_json::to_string(&q).expect("serialize half-plane point");
        let q_back: HalfPlanePoint =
            serde_json::from_str(&q_json).expect("deserialize half-plane point");
        assert_eq!(q_back, q);
        assert!(serde_json::from_str::<HalfPlanePoint>(r#"{"re":0.0,"im":0.0}"#).is_err());
        assert!(serde_json::from_str::<HalfPlanePoint>(r#"{"re":0.0,"im":-0.5}"#).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn hyperbolic_isometries_enforce_determinant_invariants_on_deserialize() {
        let psu = Psu11::translation_from_origin(DiskPoint::new(0.3, -0.2).expect("disk point"));
        let psu_json = serde_json::to_string(&psu).expect("serialize Psu11");
        let psu_back: Psu11 = serde_json::from_str(&psu_json).expect("deserialize Psu11");
        assert!((psu_back.determinant() - 1.0).abs() < 1.0e-12);
        assert!((psu_back.a() - psu.a()).norm() < 1.0e-12);
        assert!(
            serde_json::from_str::<Psu11>(r#"{"a_re":0.5,"a_im":0.0,"b_re":0.5,"b_im":0.0}"#)
                .is_err()
        );

        let psl = psu.to_psl2r();
        let psl_json = serde_json::to_string(&psl).expect("serialize Psl2R");
        let psl_back: Psl2R = serde_json::from_str(&psl_json).expect("deserialize Psl2R");
        assert!((psl_back.determinant() - 1.0).abs() < 1.0e-12);
        assert!(
            serde_json::from_str::<Psl2R>(r#"{"alpha":1.0,"beta":2.0,"gamma":2.0,"delta":4.0}"#)
                .is_err()
        );

        let table =
            BilliardTable::new(BilliardDomainKind::RegularPolygon { p: 5, q: 4 }).expect("table");
        let table_json = serde_json::to_string(&table).expect("serialize BilliardTable");
        let table_back: BilliardTable =
            serde_json::from_str(&table_json).expect("deserialize BilliardTable");
        assert_eq!(table_back, table);
        assert!(
            serde_json::from_str::<BilliardTable>(r#"{"kind":{"RegularPolygon":{"p":3,"q":3}}}"#)
                .is_err()
        );
    }
}
