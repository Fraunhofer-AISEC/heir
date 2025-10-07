use std::hint::black_box;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Stats {
    pub iterations: usize,
    pub mean_ms: f64,
    pub stddev_ms: f64,  // sample stddev
    pub median_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

pub fn compute_stats(mut samples_ms: Vec<f64>) -> Stats {
    let n = samples_ms.len();
    let mut s = Stats {
        iterations: n,
        mean_ms: 0.0,
        stddev_ms: 0.0,
        median_ms: 0.0,
        min_ms: 0.0,
        max_ms: 0.0,
    };
    if n == 0 {
        return s;
    }

    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s.min_ms = samples_ms[0];
    s.max_ms = samples_ms[n - 1];

    let sum: f64 = samples_ms.iter().copied().sum();
    s.mean_ms = sum / (n as f64);

    if n > 1 {
        let mut accum = 0.0;
        for &x in &samples_ms {
            let d = x - s.mean_ms;
            accum += d * d;
        }
        s.stddev_ms = (accum / ((n - 1) as f64)).sqrt();
    }

    s.median_ms = if n % 2 == 0 {
        0.5 * (samples_ms[n / 2 - 1] + samples_ms[n / 2])
    } else {
        samples_ms[n / 2]
    };

    s
}

pub fn print(s: &Stats, label: Option<&str>) {
    if let Some(l) = label {
        println!("{l}");
    }
    println!("runtime over {} iterations (ms):", s.iterations);
    println!("Mean:   {}", s.mean_ms);
    println!("StdDev: {}", s.stddev_ms);
    println!("Median: {}", s.median_ms);
    println!("Min:    {}", s.min_ms);
    println!("Max:    {}", s.max_ms);
}

pub fn run<F, R>(mut func: F, warmup_iters: usize, bench_iters: usize) -> Stats
where
    F: FnMut() -> R,
{
    // Warm-up
    for _ in 0..warmup_iters {
        let r = func();
        black_box(&r);
    }

    // Timed iterations
    let mut samples_ms = Vec::with_capacity(bench_iters);
    let mut last: Option<R> = None;

    for _ in 0..bench_iters {
        let t0 = Instant::now();
        let r = func();
        let dur = t0.elapsed();
        samples_ms.push(dur.as_secs_f64() * 1000.0);
        last = Some(r);
    }

    // Prevent elision of the last result for non-() types.
    if let Some(ref v) = last {
        black_box(v);
    }

    compute_stats(samples_ms)
}