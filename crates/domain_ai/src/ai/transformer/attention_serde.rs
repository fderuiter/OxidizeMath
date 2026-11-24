use super::MultiHeadAttention;
use nalgebra::DMatrix;

impl<'de> serde::Deserialize<'de> for MultiHeadAttention {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RawMultiHeadAttention {
            d_model: usize,
            h: usize,
            d_k: usize,
            w_q: DMatrix<f64>,
            w_k: DMatrix<f64>,
            w_v: DMatrix<f64>,
            w_o: DMatrix<f64>,
        }

        let raw = RawMultiHeadAttention::deserialize(deserializer)?;
        if raw.h == 0 || !raw.d_model.is_multiple_of(raw.h) || raw.d_k != raw.d_model / raw.h {
            return Err(serde::de::Error::custom(
                "MultiHeadAttention requires h > 0, d_model divisible by h, and d_k == d_model / h",
            ));
        }
        let expected_shape = (raw.d_model, raw.d_model);
        if raw.w_q.shape() != expected_shape
            || raw.w_k.shape() != expected_shape
            || raw.w_v.shape() != expected_shape
            || raw.w_o.shape() != expected_shape
        {
            return Err(serde::de::Error::custom(
                "MultiHeadAttention weight matrices must all have shape (d_model, d_model)",
            ));
        }
        Ok(Self {
            d_model: raw.d_model,
            h: raw.h,
            d_k: raw.d_k,
            w_q: raw.w_q,
            w_k: raw.w_k,
            w_v: raw.w_v,
            w_o: raw.w_o,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_multi_head_attention_serde_validation() {
        let mha = MultiHeadAttention::new(4, 2);
        let json = serde_json::to_string(&mha).unwrap();
        let decoded: MultiHeadAttention = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.d_model, 4);
        assert_eq!(decoded.h, 2);
        assert_eq!(decoded.d_k, 2);

        let bad_heads = json.replace("\"h\":2", "\"h\":0");
        assert!(serde_json::from_str::<MultiHeadAttention>(&bad_heads).is_err());

        let bad_dk = json.replace("\"d_k\":2", "\"d_k\":3");
        assert!(serde_json::from_str::<MultiHeadAttention>(&bad_dk).is_err());
    }
}
