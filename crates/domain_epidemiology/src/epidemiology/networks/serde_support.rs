use super::{NetworkEpidemicModel, NetworkEpidemicModelBuilder, NodeState};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Serialize)]
struct NetworkEpidemicModelRef<'a> {
    num_nodes: usize,
    states: &'a [NodeState],
    positions: &'a [[f32; 2]],
    adjacency: &'a [Vec<usize>],
    edge_quarantined: &'a [Vec<bool>],
    patient_zero: Option<usize>,
    beta: f64,
    gamma: f64,
}

#[derive(Deserialize)]
struct RawNetworkEpidemicModel {
    num_nodes: usize,
    states: Vec<NodeState>,
    positions: Vec<[f32; 2]>,
    adjacency: Vec<Vec<usize>>,
    #[serde(default)]
    edge_quarantined: Option<Vec<Vec<bool>>>,
    #[serde(default)]
    patient_zero: Option<Option<usize>>,
    beta: f64,
    gamma: f64,
}

fn apply_quarantine_masks(
    model: &mut NetworkEpidemicModel,
    masks: Vec<Vec<bool>>,
) -> Result<(), &'static str> {
    if masks.len() != model.num_nodes {
        return Err("edge_quarantined length must match num_nodes");
    }
    for (u, row) in masks.iter().enumerate() {
        if row.len() != model.adjacency[u].len() {
            return Err("edge_quarantined row length must match adjacency row length");
        }
        for (idx, &q) in row.iter().enumerate() {
            if q {
                let v = model.adjacency[u][idx];
                let sym = model.adjacency[v]
                    .iter()
                    .position(|&nbr| nbr == u)
                    .and_then(|rev| masks.get(v)?.get(rev).copied())
                    .unwrap_or(false);
                if !sym {
                    return Err("edge_quarantined masks must be symmetric across undirected edges");
                }
                model
                    .set_edge_quarantined(u, v, true)
                    .map_err(|_| "Invalid quarantined edge in NetworkEpidemicModel")?;
            }
        }
    }
    Ok(())
}

impl Serialize for NetworkEpidemicModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        NetworkEpidemicModelRef {
            num_nodes: self.num_nodes,
            states: &self.states,
            positions: &self.positions,
            adjacency: &self.adjacency,
            edge_quarantined: &self.edge_quarantined,
            patient_zero: self.patient_zero,
            beta: self.beta,
            gamma: self.gamma,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for NetworkEpidemicModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawNetworkEpidemicModel::deserialize(deserializer)?;
        if raw
            .positions
            .iter()
            .any(|p| !p[0].is_finite() || !p[1].is_finite())
        {
            return Err(serde::de::Error::custom(
                "Node positions must be finite floats",
            ));
        }
        if raw
            .adjacency
            .iter()
            .any(|neighbors| neighbors.iter().any(|&idx| idx >= raw.num_nodes))
        {
            return Err(serde::de::Error::custom(
                "Adjacency indices must be < num_nodes",
            ));
        }
        let mut model = NetworkEpidemicModelBuilder::new()
            .num_nodes(raw.num_nodes)
            .parameters(raw.beta, raw.gamma)
            .states(raw.states)
            .positions(raw.positions)
            .adjacency(raw.adjacency)
            .build()
            .map_err(serde::de::Error::custom)?;
        if let Some(masks) = raw.edge_quarantined {
            apply_quarantine_masks(&mut model, masks).map_err(serde::de::Error::custom)?;
        }
        if let Some(pz_opt) = raw.patient_zero {
            if let Some(pz) = pz_opt
                && pz >= model.num_nodes
            {
                return Err(serde::de::Error::custom("patient_zero index out of bounds"));
            }
            model.patient_zero = pz_opt;
        }
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{NetworkEpidemicModel, NodeState};

    #[test]
    #[verified_engine::verified]
    fn test_network_epidemic_model_serde_roundtrip_and_validation() {
        let mut model = NetworkEpidemicModel::builder()
            .num_nodes(3)
            .parameters(0.4, 0.1)
            .states(vec![
                NodeState::Infected,
                NodeState::Vaccinated,
                NodeState::Susceptible,
            ])
            .positions(vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
            .adjacency(vec![vec![1], vec![0, 2], vec![1]])
            .build()
            .expect("valid chain model");
        model
            .set_edge_quarantined(1, 2, true)
            .expect("quarantine edge");

        let json = serde_json::to_string(&model).expect("serialize model");
        let restored: NetworkEpidemicModel =
            serde_json::from_str(&json).expect("deserialize model");
        assert_eq!(restored, model);
        assert!(restored.is_edge_quarantined(1, 2));

        let bad_beta = r#"{"num_nodes":1,"states":["Susceptible"],"positions":[[0.0,0.0]],"adjacency":[[]],"beta":-0.1,"gamma":0.1}"#;
        assert!(serde_json::from_str::<NetworkEpidemicModel>(bad_beta).is_err());
    }
}
