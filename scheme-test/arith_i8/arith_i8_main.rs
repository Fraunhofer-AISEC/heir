// src/main.rs
use tfhe::prelude::*;
use tfhe::{generate_keys, set_server_key, ConfigBuilder, FheUint8};

// Generated module
use heir_tfhe_project::generated;

mod bench;

fn main() {
    // 1) Keygen for 32-bit typed integers
    let config = ConfigBuilder::default().build();
    let (client_key, server_key) = generate_keys(config);

    // 2) Make the server key globally available for high-level ops
    set_server_key(server_key);

    // 3) Encrypt input(s) once, reused across all iterations
    let x1 = FheUint8::encrypt(42u8, &client_key);
    let x2 = FheUint8::encrypt(42u8, &client_key);

    println!("Starting position");

    // Optional: cold run to trigger any lazy init before timing
    {
        let _ = generated::foo(&x1, &x2);
    }

    println!("After cold run");

    // 4) Benchmark only the foo operation (exclude encryption/decryption)
    let warmup_iters = 5;
    let bench_iters = 10;

    println!("Before benchmarking");

    let stats = bench::run(
        || {
            // Measure the homomorphic function call only
            generated::foo(&x1, &x2)
        },
        warmup_iters,
        bench_iters,
    );

    println!("After benchmarking");

    bench::print(&stats, Some("foo() timing"));

    // 5) Verify the result (not part of timing)
    let y = generated::foo(&x1, &x2);
    let y_clear: u8 = y.decrypt(&client_key);
    println!("foo(42, 42) = {}", y_clear);
}