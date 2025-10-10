// src/main.rs
use tfhe::shortint::parameters::PARAM_MESSAGE_3_CARRY_3;
use tfhe::shortint::{gen_keys, ClientKey, Ciphertext, ServerKey};

use heir_tfhe_project::generated;

mod bench;

/// Encrypt a u32 as 32 shortint bit-ciphertexts (LSB first) 
fn encrypt_bits(client_key: &ClientKey, value: u32) -> [Ciphertext; 32] {
    core::array::from_fn(|i| {
        let bit = ((value >> i) & 1) as u64;
        client_key.encrypt(bit)
    })
}

/// Decrypt 32 shortint bit-ciphertexts back into a u32 (LSB first)
fn decrypt_bits_to_u32(client_key: &ClientKey, bits: &[Ciphertext; 32]) -> u32 {
    let mut out = 0u32;
    for i in 0..32 {
        let m = client_key.decrypt(&bits[i]) as u32;
        out |= (m & 1) << i;
    }
    out
}

fn main() {
    // Use a shortint parameter that can represent values up to 7 (due to shifts/adds in foo)
    let (client_key, server_key) = gen_keys(PARAM_MESSAGE_3_CARRY_3);

    // Encrypt inputs as 32-bit arrays
    let x1_val: u32 = 2;
    let x2_val: u32 = 3;
    let x1_bits = encrypt_bits(&client_key, x1_val);
    let x2_bits = encrypt_bits(&client_key, x2_val);

    println!("Starting position");

    // Optional: cold run to trigger any lazy init before timing
    {
        let _ = generated::foo(&server_key, &x1_bits, &x2_bits);
    }

    println!("After cold run");

    // Benchmark only the foo operation (exclude encryption/decryption)
    let warmup_iters = 5;
    let bench_iters = 10;

    println!("Before benchmarking");

    let stats = bench::run(
        || {
            // Measure the homomorphic function call only
            generated::foo(&server_key, &x1_bits, &x2_bits)
        },
        warmup_iters,
        bench_iters,
    );

    println!("After benchmarking");

    bench::print(&stats, Some("foo() timing"));

    // Verify the result (not part of timing)
    let y_bits = generated::foo(&server_key, &x1_bits, &x2_bits);
    let y_clear = decrypt_bits_to_u32(&client_key, &y_bits);
    println!("foo({}, {}) = {}", x1_val, x2_val, y_clear);
}