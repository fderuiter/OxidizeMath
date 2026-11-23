use nalgebra::DMatrix;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A basic tokenizer interface.
pub trait Tokenizer {
    /// Tokenizes the input text into a list of string tokens.
    #[verified_engine::verified]
    fn tokenize(&self, text: &str) -> Vec<String>;
}

/// A basic whitespace and punctuation tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WordTokenizer;

impl Tokenizer for WordTokenizer {
    #[verified_engine::verified]
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();

        for c in text.chars() {
            if c.is_alphanumeric() {
                current_token.push(c);
            } else {
                if !current_token.is_empty() {
                    tokens.push(current_token.clone());
                    current_token.clear();
                }
                if !c.is_whitespace() {
                    tokens.push(c.to_string());
                }
            }
        }
        if !current_token.is_empty() {
            tokens.push(current_token);
        }

        tokens
    }
}

/// A naive character-level tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CharTokenizer;

impl Tokenizer for CharTokenizer {
    #[verified_engine::verified]
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| c.to_string())
            .collect()
    }
}

/// An Embedding Matrix generator for visualization purposes.
/// In a real system, this would be a learnable parameter matrix `[vocab_size, d_model]`.
/// Here, we use a deterministic hash to generate a pseudo-embedding for any token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PseudoEmbedding {
    #[allow(missing_docs)]
    pub d_model: usize,
}

impl PseudoEmbedding {
    /// Generates a pseudo-embedding vector for a single token.
    #[verified_engine::verified]
    pub fn embed_token(&self, token: &str) -> Vec<f64> {
        let mut hasher = DefaultHasher::new();
        token.hash(&mut hasher);
        let base_hash = hasher.finish();

        let mut embedding = Vec::with_capacity(self.d_model);
        for i in 0..self.d_model {
            // Mix the hash with the dimension index to get pseudo-random but deterministic values
            let val = ((base_hash.wrapping_add(i as u64)) % 2000) as f64 / 1000.0 - 1.0;
            embedding.push(val);
        }
        embedding
    }

    /// Embeds a sequence of tokens into a `DMatrix<f64>` of shape `(seq_len, d_model)`.
    #[verified_engine::verified]
    pub fn embed_sequence(&self, tokens: &[String]) -> DMatrix<f64> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return DMatrix::zeros(0, self.d_model);
        }

        let mut matrix = DMatrix::zeros(seq_len, self.d_model);
        for (i, token) in tokens.iter().enumerate() {
            let emb = self.embed_token(token);
            for j in 0..self.d_model {
                matrix[(i, j)] = emb[j];
            }
        }
        matrix
    }
}
