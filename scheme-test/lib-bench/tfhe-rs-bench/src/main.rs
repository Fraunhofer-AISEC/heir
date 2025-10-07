use std::env;
use std::hint::black_box;
use std::str::FromStr;
use std::time::Instant;

use tfhe::{ClientKey, ServerKey, ConfigBuilder, generate_keys, set_server_key};
use tfhe::{FheUint8, FheUint16, FheUint32, FheUint64, FheUint128};
use tfhe::prelude::*;

// Simple CLI parsing
#[derive(Clone)]
struct Cli {
    bits: Vec<u32>, // selectable bit-widths
    warmup: usize,
    iters: usize,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            bits: vec![8, 16, 32, 64, 128],
            warmup: 5,
            iters: 50,
        }
    }
}

fn split_ints<T: FromStr>(s: &str) -> Vec<T> {
    s.split(',').filter_map(|x| x.parse::<T>().ok()).collect()
}

fn parse_cli() -> Cli {
    let mut cli = Cli::default();
    for a in env::args().skip(1) {
        if let Some(rest) = a.strip_prefix("--bits=") {
            let v = split_ints::<u32>(rest);
            let mut v2: Vec<u32> = v
                .into_iter()
                .filter(|b| matches!(*b, 8 | 16 | 32 | 64 | 128))
                .collect();
            v2.sort_unstable();
            v2.dedup();
            if !v2.is_empty() {
                cli.bits = v2;
            }
        } else if let Some(rest) = a.strip_prefix("--warmup=") {
            if let Ok(w) = rest.parse::<usize>() {
                cli.warmup = w;
            }
        } else if let Some(rest) = a.strip_prefix("--iters=") {
            if let Ok(i) = rest.parse::<usize>() {
                cli.iters = i;
            }
        } else if a == "--help" || a == "-h" {
            println!("Usage: tfhe_bench [--bits=8,16,32,64,128] [--warmup=5] [--iters=50]");
            std::process::exit(0);
        }
    }
    cli
}

#[derive(Clone, Debug)]
struct Stats {
    mean_ms: f64,
    stddev_ms: f64,
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
    iterations: usize,
}

fn compute_stats(samples_ms: &[f64]) -> Stats {
    let n = samples_ms.len();
    let iterations = n;
    let min_ms = samples_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ms = samples_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean_ms = if n == 0 {
        0.0
    } else {
        samples_ms.iter().copied().sum::<f64>() / n as f64
    };
    let mut sorted = samples_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = if n == 0 {
        0.0
    } else if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    };
    let var = if n <= 1 {
        0.0
    } else {
        let mut acc = 0.0;
        for &x in samples_ms {
            let d = x - mean_ms;
            acc += d * d;
        }
        acc / (n as f64)
    };
    let stddev_ms = var.sqrt();
    Stats {
        mean_ms,
        stddev_ms,
        median_ms,
        min_ms,
        max_ms,
        iterations,
    }
}

fn run<T, F: FnMut() -> T>(mut op: F, warmup: usize, iters: usize) -> Stats {
    for _ in 0..warmup {
        black_box(op());
    }
    let mut samples_ms = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let res = op();
        black_box(res);
        samples_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    compute_stats(&samples_ms)
}

#[derive(Clone)]
struct Row {
    bits: u32,
    add: Stats,
    sub: Stats,
    mul: Stats,
}

fn print_table(title: &str, rows: &[Row], which: char) {
    println!("\n{} (ms)", title);
    println!("| bits | mean | stddev | median | min | max | iters |");
    println!("|-----:|-----:|-------:|-------:|----:|----:|------:|");
    for r in rows {
        let s = match which {
            'a' => &r.add,
            's' => &r.sub,
            'm' => &r.mul,
            _ => &r.mul,
        };
        println!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} |",
            r.bits, s.mean_ms, s.stddev_ms, s.median_ms, s.min_ms, s.max_ms, s.iterations
        );
    }
}

fn setup_keys(enabled_bits: &[u32]) -> (ClientKey, ServerKey) {
    let mut builder = ConfigBuilder::default();
    let config = builder.build();
    let (client_key, server_key) = generate_keys(config);
    // Set global server key so +, -, * work on ciphertexts
    set_server_key(server_key.clone());
    (client_key, server_key)
}

fn main() {
    let cli = parse_cli();

    let (client_key, _server_key) = setup_keys(&cli.bits);

    let mut results: Vec<Row> = Vec::with_capacity(cli.bits.len());

    for bits in &cli.bits {
        println!(
            "TFHE integer: bits = {}, warmup = {}, iters = {}",
            bits, cli.warmup, cli.iters
        );

        match bits {
            8 => {
                let a: u8 = 123;
                let b: u8 = 45;
                let ct_a = FheUint8::encrypt(a, &client_key);
                let ct_b = FheUint8::encrypt(b, &client_key);

                let add = run(|| &ct_a + &ct_b, cli.warmup, cli.iters);
                let sub = run(|| &ct_a - &ct_b, cli.warmup, cli.iters);
                let mul = run(|| &ct_a * &ct_b, cli.warmup, cli.iters);

                results.push(Row { bits: *bits, add, sub, mul });
            }
            16 => {
                let a: u16 = 12_345;
                let b: u16 = 2_345;
                let ct_a = FheUint16::encrypt(a, &client_key);
                let ct_b = FheUint16::encrypt(b, &client_key);

                let add = run(|| &ct_a + &ct_b, cli.warmup, cli.iters);
                let sub = run(|| &ct_a - &ct_b, cli.warmup, cli.iters);
                let mul = run(|| &ct_a * &ct_b, cli.warmup, cli.iters);

                results.push(Row { bits: *bits, add, sub, mul });
            }
            32 => {
                let a: u32 = 123_456_789;
                let b: u32 = 42_000;
                let ct_a = FheUint32::encrypt(a, &client_key);
                let ct_b = FheUint32::encrypt(b, &client_key);

                let add = run(|| &ct_a + &ct_b, cli.warmup, cli.iters);
                let sub = run(|| &ct_a - &ct_b, cli.warmup, cli.iters);
                let mul = run(|| &ct_a * &ct_b, cli.warmup, cli.iters);

                results.push(Row { bits: *bits, add, sub, mul });
            }
            64 => {
                let a: u64 = 1_234_567_890_123;
                let b: u64 = 4_200_000_000;
                let ct_a = FheUint64::encrypt(a, &client_key);
                let ct_b = FheUint64::encrypt(b, &client_key);

                let add = run(|| &ct_a + &ct_b, cli.warmup, cli.iters);
                let sub = run(|| &ct_a - &ct_b, cli.warmup, cli.iters);
                let mul = run(|| &ct_a * &ct_b, cli.warmup, cli.iters);

                results.push(Row { bits: *bits, add, sub, mul });
            }
            128 => {
                let a: u128 = 123_456_789_012_345_678_901_234_567_890u128;
                let b: u128 = 42_000_000_000_000_000_000u128;
                let ct_a = FheUint128::encrypt(a, &client_key);
                let ct_b = FheUint128::encrypt(b, &client_key);

                let add = run(|| &ct_a + &ct_b, cli.warmup, cli.iters);
                let sub = run(|| &ct_a - &ct_b, cli.warmup, cli.iters);
                let mul = run(|| &ct_a * &ct_b, cli.warmup, cli.iters);

                results.push(Row { bits: *bits, add, sub, mul });
            }
            _ => {
                eprintln!("Unsupported bit-width: {} (allowed: 8,16,32,64,128)", bits);
            }
        }
    }

    print_table("TFHE Integer Add", &results, 'a');
    print_table("TFHE Integer Sub", &results, 's');
    print_table("TFHE Integer Mul", &results, 'm');
}