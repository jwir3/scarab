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

    let batch_size = 100;
    // Construct the initial conditions outside the benchmark loops
    let initial_conditions = vec![(1.0, 45.0); batch_size];

    for steps in [1_000, 10_000] {
        // Optional: Reports throughput in total trajectory steps/sec across the batch
        group.throughput(Throughput::Elements((steps * batch_size as u64) as u64));

        // 1. Sequential Baseline
        // Updated to process the full batch sequentially for an apples-to-apples comparison
        group.bench_with_input(BenchmarkId::new("sequential", steps), &steps, |b, &s| {
            let mut guide = Waveguide::new(1.0, 0.001, 1.0, 2);
            b.iter(|| {
                for (x_start, theta_start) in initial_conditions.iter() {
                    guide.compute_trajectory(
                        black_box(*x_start),
                        black_box(*theta_start),
                        black_box(s),
                    );
                }
            });
        });

        // 2. Parallelized Version
        group.bench_with_input(BenchmarkId::new("parallel", steps), &steps, |b, &s| {
            let guide = Waveguide::new(1.0, 0.001, 1.0, 2);

            b.iter(|| {
                guide.compute_multiple_trajectories(
                    black_box(initial_conditions.clone()),
                    black_box(s),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_waveguide_comparison);
criterion_main!(benches);
