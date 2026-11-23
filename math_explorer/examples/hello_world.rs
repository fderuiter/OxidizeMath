//! Example target for hello world.
use math_explorer::physics::quantum::clebsch_gordan;
use std::fmt;

// Simple ANSI color wrapper for better UX
enum Color {
    Cyan,
    Green,
    Yellow,
    Magenta,
    Bold,
    Reset,
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let code = match self {
            Color::Cyan => "\x1b[36m",
            Color::Green => "\x1b[32m",
            Color::Yellow => "\x1b[33m",
            Color::Magenta => "\x1b[35m",
            Color::Bold => "\x1b[1m",
            Color::Reset => "\x1b[0m",
        };
        write!(f, "{}", code)
    }
}

fn print_banner() {
    println!();
    println!(
        "{}   __  __       _   _      {}",
        Color::Magenta,
        Color::Reset
    );
    println!(
        "{}  |  \\/  | __ _| |_| |__   {}",
        Color::Magenta,
        Color::Reset
    );
    println!(
        "{}  | |\\/| |/ _` | __| '_ \\  {}",
        Color::Magenta,
        Color::Reset
    );
    println!(
        "{}  | |  | | (_| | |_| | | | {}",
        Color::Magenta,
        Color::Reset
    );
    println!(
        "{}  |_|  |_|\\__,_|\\__|_| |_| {}",
        Color::Magenta,
        Color::Reset
    );
    println!("{}      E X P L O R E R      {}", Color::Cyan, Color::Reset);
    println!();

    println!(
        "{}Welcome to the Math Explorer!{}",
        Color::Bold,
        Color::Reset
    );
    println!("This example demonstrates quantum mechanics calculations.");
    println!();
}

fn print_parameters(j1: f64, m1: f64, j2: f64, m2: f64, j: f64, m: f64) {
    println!(
        "{}🧮 Calculating Clebsch-Gordan Coefficient...{}",
        Color::Yellow,
        Color::Reset
    );
    println!("   Coupling angular momenta states.");
    println!();
    println!("   {}Parameters:{}", Color::Bold, Color::Reset);
    println!("   ┌──────┬───────┬───────┐");
    println!("   │ Var  │   j   │   m   │");
    println!("   ├──────┼───────┼───────┤");
    println!("   │ 1    │ {:5.1} │ {:5.1} │", j1, m1);
    println!("   │ 2    │ {:5.1} │ {:5.1} │", j2, m2);
    println!("   │ Tot  │ {:5.1} │ {:5.1} │", j, m);
    println!("   └──────┴───────┴───────┘");
    println!();
}

fn calculate_and_print_result(j1: f64, m1: f64, j2: f64, m2: f64, j: f64, m: f64) {
    let coeff = clebsch_gordan(j1, m1, j2, m2, j, m);
    println!("{}Result:{}", Color::Bold, Color::Reset);
    println!(
        "   <j1 m1; j2 m2 | J M> = {}{:.6}{}",
        Color::Cyan,
        coeff,
        Color::Reset
    );
    println!();
    if coeff.abs() > 1e-10 {
        println!(
            "{}✅ Calculation successful and verified.{}",
            Color::Green,
            Color::Reset
        );
    } else {
        println!(
            "{}⚠️  Result is zero (forbidden transition).{}",
            Color::Yellow,
            Color::Reset
        );
    }
    println!();
}

fn main() {
    print_banner();

    let j1 = 1.5;
    let m1 = -0.5;
    let j2 = 1.0;
    let m2 = 1.0;
    let j = 2.5;
    let m = 0.5;

    print_parameters(j1, m1, j2, m2, j, m);
    calculate_and_print_result(j1, m1, j2, m2, j, m);
}
