use crate::async_sim::unified::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::StateSnapshot;
use scientific_metadata::theory::ParameterConstraint;
use math_explorer::biology::neuroscience::{
    HodgkinHuxleyNeuron, HodgkinHuxleyParameters,
};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

#[cfg_attr(any(), verified(opt_out = "gui_tool"))]
pub struct HodgkinHuxleyUnified {
    neuron: HodgkinHuxleyNeuron,
    params: HodgkinHuxleyParameters,
    history: VecDeque<[f64; 2]>,
    time: f64,
    i_ext: f64,
    dt: f64,
}

impl UnifiedModel for HodgkinHuxleyUnified {
    fn new(params: &HashMap<String, f64>) -> Self {
        let g_na = *params.get("g_na").unwrap_or(&120.0);
        let g_k = *params.get("g_k").unwrap_or(&36.0);
        let g_l = *params.get("g_l").unwrap_or(&0.3);
        let i_ext = *params.get("i_ext").unwrap_or(&10.0);
        
        let hh_params = HodgkinHuxleyParameters {
            g_na,
            g_k,
            g_l,
            ..Default::default()
        };

        Self {
            neuron: HodgkinHuxleyNeuron::builder().with_params(hh_params.clone()).build().unwrap_or(HodgkinHuxleyNeuron::new(-65.0)),
            params: hh_params,
            history: VecDeque::new(),
            time: 0.0,
            i_ext,
            dt: 0.01,
        }
    }

    fn step(&mut self, params: &HashMap<String, f64>) {
        let new_g_na = *params.get("g_na").unwrap_or(&120.0);
        let new_g_k = *params.get("g_k").unwrap_or(&36.0);
        let new_g_l = *params.get("g_l").unwrap_or(&0.3);
        let new_i_ext = *params.get("i_ext").unwrap_or(&10.0);

        let params_changed = self.params.g_na != new_g_na || 
                             self.params.g_k != new_g_k || 
                             self.params.g_l != new_g_l;

        self.params.g_na = new_g_na;
        self.params.g_k = new_g_k;
        self.params.g_l = new_g_l;
        self.i_ext = new_i_ext;

        if params_changed {
            let builder = HodgkinHuxleyNeuron::builder()
                .with_initial_v(self.neuron.v())
                .with_n(self.neuron.n())
                .with_m(self.neuron.m())
                .with_h(self.neuron.h())
                .with_params(self.params.clone());

            if let Ok(new_neuron) = builder.build() {
                self.neuron = new_neuron;
            }
        }

        self.neuron.update(self.dt, self.i_ext);
        self.time += self.dt;
        self.history.push_back([self.time, self.neuron.v()]);
        while self.history.len() > 10_000 {
            self.history.pop_front();
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let mut custom_data = Vec::with_capacity(self.history.len() * 2 + 2);
        for &[t, v] in &self.history {
            custom_data.push(t);
            custom_data.push(v);
        }
        custom_data.push(self.time);
        custom_data.push(self.neuron.v());

        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(std::sync::RwLock::new(Vec::new())),
            custom_data,
            structured_data: None,
        }
    }

    fn parameters() -> HashMap<String, ParameterConstraint> {
        let mut map = HashMap::new();
        map.insert("g_na".to_string(), ParameterConstraint { min: 0.0, max: 200.0, step: 1.0 });
        map.insert("g_k".to_string(), ParameterConstraint { min: 0.0, max: 100.0, step: 1.0 });
        map.insert("g_l".to_string(), ParameterConstraint { min: 0.0, max: 5.0, step: 0.1 });
        map.insert("i_ext".to_string(), ParameterConstraint { min: 0.0, max: 50.0, step: 1.0 });
        map
    }

    fn name() -> &'static str {
        "Hodgkin-Huxley Model"
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        name: "HodgkinHuxleyTool",
        domain: "neuroscience",
        tags: &[],
        build: || Box::new(UnifiedSimTool::<HodgkinHuxleyUnified>::new()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for HodgkinHuxleyUnified {
    fn theory_description(&self) -> String {
        math_explorer::biology::neuroscience::HodgkinHuxleyModel::new(math_explorer::biology::neuroscience::HodgkinHuxleyParameters::default(), 0.0).theory_description()
    }
    
    fn phonetic_description(&self) -> String {
        math_explorer::biology::neuroscience::HodgkinHuxleyModel::new(math_explorer::biology::neuroscience::HodgkinHuxleyParameters::default(), 0.0).phonetic_description()
    }
    
    fn theory_citation(&self) -> String {
        math_explorer::biology::neuroscience::HodgkinHuxleyModel::new(math_explorer::biology::neuroscience::HodgkinHuxleyParameters::default(), 0.0).theory_citation()
    }
    
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        math_explorer::biology::neuroscience::HodgkinHuxleyModel::new(math_explorer::biology::neuroscience::HodgkinHuxleyParameters::default(), 0.0).available_descriptions()
    }
}
