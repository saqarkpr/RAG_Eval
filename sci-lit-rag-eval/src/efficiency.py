"""CPU-only efficiency comparison between the two retrievers: index build
time, per-query latency (mean/p95), and approximate memory footprint. This
stands in for the 'efficient methods' axis of the evaluation when no GPU or
quantized-LLM inference is available in the environment (see README
Limitations)."""
import time
import numpy as np


def benchmark_retriever(retriever, queries, n_repeats=3, k=5):
    latencies = []
    for _ in range(n_repeats):
        for q in queries:
            t0 = time.perf_counter()
            retriever.retrieve(q, k=k)
            latencies.append(time.perf_counter() - t0)
    latencies = np.array(latencies)
    return {
        "build_time_s": retriever.build_time_s,
        "mean_latency_ms": float(latencies.mean() * 1000),
        "p95_latency_ms": float(np.percentile(latencies, 95) * 1000),
        "approx_memory_kb": retriever.approx_memory_bytes() / 1024.0,
        "n_queries_timed": len(latencies),
    }
