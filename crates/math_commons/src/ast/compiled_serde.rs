//! Validated `serde::Deserialize` implementation for [`CompiledExpr`] postfix bytecode.
// [cite:math_commons]

use super::{CompiledExpr, Instruction, verify_stack_depth};
use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Deserializer};

#[derive(Deserialize)]
#[serde(rename = "CompiledExpr")]
struct CompiledExprWire {
    instructions: Vec<Instruction>,
    var_names: Vec<String>,
    axis_count: usize,
}

fn validate_wire(wire: &CompiledExprWire) -> Result<(), &'static str> {
    if wire.axis_count > wire.var_names.len() {
        return Err("CompiledExpr axis_count exceeds var_names length");
    }
    if wire.var_names.len() > usize::from(u16::MAX) {
        return Err("CompiledExpr var_names exceeds u16 slot capacity");
    }
    if wire.instructions.is_empty() {
        return Err("CompiledExpr instructions cannot be empty");
    }
    verify_stack_depth(&wire.instructions)
        .map_err(|_| "CompiledExpr instructions violate stack depth or instruction budget")?;

    let mut sp = 0usize;
    for &inst in &wire.instructions {
        match inst {
            Instruction::PushConst(_) | Instruction::PushImagUnit => sp += 1,
            Instruction::LoadVar(slot) => {
                if usize::from(slot) >= wire.var_names.len() {
                    return Err("CompiledExpr LoadVar slot out of bounds for var_names");
                }
                sp += 1;
            }
            Instruction::Unary(_) => {}
            Instruction::Binary(_) => sp -= 1,
            Instruction::Call(func) => sp = sp + 1 - func.arity(),
        }
    }
    if sp != 1 {
        return Err("CompiledExpr instructions must leave exactly 1 value on the stack");
    }
    Ok(())
}

impl<'de> Deserialize<'de> for CompiledExpr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CompiledExprWire::deserialize(deserializer)?;
        validate_wire(&wire).map_err(serde::de::Error::custom)?;
        Ok(Self {
            instructions: wire.instructions,
            var_names: wire.var_names,
            axis_count: wire.axis_count,
        })
    }
}
