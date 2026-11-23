//! Example target for transformer demo.
use math_explorer::ai::transformer::Encoder;
use math_explorer::nalgebra::DMatrix;

fn main() {
    // Initialize an Encoder stack: 2 layers, 512 embedding dim, 8 heads, 2048 FF dim
    let encoder = Encoder::new(2, 512, 8, 2048);

    // Dummy input: Sequence length 10
    let input = DMatrix::zeros(10, 512);
    let _encoded = encoder.forward(input, None);
}
