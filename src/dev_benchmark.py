"""Measure deterministic Aether frontend and Tri-View latency/memory."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
import time
import tracemalloc

from src.ai_view import (
    create_encoder,
    default_profile,
    generate_ai_view_with_map,
)
from src.codemap import build_codemap, render_map
from src.frontend import parse_source


def _percentile(samples: list[int], percentile: float) -> int:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1))
    return ordered[index]


def measure(
    source: str,
    iterations: int = 100,
    profile=None,
    encoder=None,
) -> dict[str, int | float]:
    """Return frontend/translation latency percentiles and peak traced memory."""
    if iterations < 1:
        raise ValueError("iterations must be at least one")
    profile = profile or default_profile()
    tokenizer_optimized = encoder is not None

    def translate() -> None:
        frontend = parse_source(source)
        errors = [item for item in frontend.diagnostics if item.severity == "error"]
        if errors:
            raise ValueError(
                "; ".join(f"{item.code}: {item.message}" for item in errors)
            )
        generate_ai_view_with_map(frontend.cst, profile, encoder)
        render_map(build_codemap(frontend.ast))

    translate()
    timings = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        translate()
        timings.append(time.perf_counter_ns() - start)

    tracemalloc.start()
    try:
        translate()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "iterations": iterations,
        "tokenizer_optimized": tokenizer_optimized,
        "median_ms": statistics.median(timings) / 1_000_000,
        "p95_ms": _percentile(timings, 0.95) / 1_000_000,
        "max_ms": max(timings) / 1_000_000,
        "peak_kib": peak_bytes / 1024,
        "source_bytes": len(source.encode("utf-8")),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "examples" / "sensor.hl",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--tokenizer-optimized",
        action="store_true",
        help="Also optimize aliases with the selected profile tokenizer (slower)",
    )
    args = parser.parse_args()
    with args.source.open("r", encoding="utf-8", newline="") as source_file:
        source = source_file.read()
    profile = default_profile()
    encoder = create_encoder(profile) if args.tokenizer_optimized else None
    result = measure(source, args.iterations, profile, encoder)
    mode = "tokenizer-optimized" if result["tokenizer_optimized"] else "offline deterministic"
    print(f"Profile: {profile.name}; mode: {mode}")
    print(f"Input: {args.source} ({result['source_bytes']} UTF-8 bytes)")
    print(f"Iterations: {result['iterations']}")
    print(f"Median: {result['median_ms']:.3f} ms")
    print(f"P95: {result['p95_ms']:.3f} ms")
    print(f"Maximum: {result['max_ms']:.3f} ms")
    print(f"Peak traced Python allocation: {result['peak_kib']:.1f} KiB")
    print("Measurements are environment-dependent and are not performance guarantees.")


if __name__ == "__main__":
    main()
