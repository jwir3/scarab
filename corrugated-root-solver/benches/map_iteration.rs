use corrugated_root_solver::Waveguide;
use criterion::{
    BenchmarkId, Criterion, SamplingMode, Throughput, black_box, criterion_group, criterion_main,
};
use std::time::Duration;

fn benchmark_waveguide_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_trajectory (1.0, 45 deg)");

    // Flat sampling is ideal when single runs take several milliseconds
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    for steps in [1_000, 10_000] {
        // Optional: Reports throughput in trajectory steps/sec
        group.throughput(Throughput::Elements(steps as u64));

        // 1. Sequential Baseline
        group.bench_with_input(BenchmarkId::new("sequential", steps), &steps, |b, &s| {
            let mut guide = Waveguide::new(1.0, 0.001, 1.0, 2);
            b.iter(|| guide.compute_trajectory(black_box(1.0), black_box(45.0), black_box(s)));
        });

        // 2. Parallelized Version (uncomment when ready)
        /*
        group.bench_with_input(BenchmarkId::new("parallel", steps), &steps, |b, &s| {
            let mut guide = Waveguide::new(1.0, 0.001, 1.0, 2);
            b.iter(|| {
                guide.compute_trajectory_parallel(black_box(1.0), black_box(45.0), black_box(s))
            });
        });
        */
    }

    group.finish();
}

criterion_group!(benches, benchmark_waveguide_comparison);
criterion_main!(benches);
