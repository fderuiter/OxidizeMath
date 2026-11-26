//! Example target for transformer demo.
use oxidizemath::ai::transformer::Encoder;
use oxidizemath::nalgebra::DMatrix;

fn main() {
    let smoke = std::env::var("MATH_EXPLORER_SMOKE")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    let (d_model, heads, d_ff) = if smoke { (64, 4, 128) } else { (512, 8, 2048) };
    let encoder = Encoder::new(2, d_model, heads, d_ff);

    // Dummy input: Sequence length 10
    let input = DMatrix::zeros(10, d_model);
    let _encoded = encoder.forward(input, None);
}
