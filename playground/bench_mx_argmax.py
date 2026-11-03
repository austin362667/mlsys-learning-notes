"""Compare argmax + gather with argmax + max on one resident GPU input.

Benchmark:
    uv run python profile_max.py
Capture warmed calls after benchmarking:
    MTL_CAPTURE_ENABLED=1 uv run python profile_max.py --capture
"""

import argparse
import os
from pathlib import Path
import random
import statistics
import tempfile
import time

import mlx.core as mx


def compute_max_single_call(logits, axis):
    max_idx = mx.stop_gradient(mx.argmax(logits, axis=axis, keepdims=True))
    max_logit = mx.stop_gradient(mx.take_along_axis(logits, max_idx, axis=axis))
    return max_idx, max_logit


def compute_max_two_calls(logits, axis):
    max_idx = mx.stop_gradient(mx.argmax(logits, axis=axis, keepdims=True))
    max_logit = mx.stop_gradient(mx.max(logits, axis=axis, keepdims=True))
    return max_idx, max_logit


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def capture(fn, logits, axis, path):
    mx.synchronize()
    mx.metal.start_capture(str(path))
    try:
        # Create Metal encoding objects within the capture scope.
        with mx.stream(mx.new_stream(mx.gpu)):
            outputs = fn(logits, axis)
            mx.eval(*outputs)
            mx.synchronize()
    finally:
        mx.metal.stop_capture()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=positive_int, default=1_000_000)
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument("--rounds", type=positive_int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--capture-dir", type=Path, default=Path("max_captures"))
    args = parser.parse_args()
    if args.capture and os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        parser.error("--capture requires MTL_CAPTURE_ENABLED=1")

    mx.set_default_device(mx.gpu)
    mx.random.seed(args.seed)
    logits = mx.random.normal((args.size,))
    axis = 0
    mx.eval(logits)
    mx.synchronize()

    variants = {
        "argmax + gather": compute_max_single_call,
        "argmax + gather (compiled)": mx.compile(compute_max_single_call),
        "argmax + max": compute_max_two_calls,
        "argmax + max (compiled)": mx.compile(compute_max_two_calls),
    }
    reference = compute_max_two_calls(logits, axis)
    mx.eval(*reference)
    for name, fn in variants.items():
        outputs = fn(logits, axis)
        mx.eval(*outputs)
        if not all(mx.array_equal(a, b).item() for a, b in zip(outputs, reference)):
            raise RuntimeError(f"Output mismatch: {name}")
        for _ in range(args.warmup):
            mx.eval(*fn(logits, axis))
    mx.synchronize()

    # Shuffle each round to reduce bias from execution order.
    order_rng = random.Random(args.seed)
    samples = {name: [] for name in variants}
    for _ in range(args.rounds):
        order = list(variants)
        order_rng.shuffle(order)
        for name in order:
            fn = variants[name]
            mx.synchronize()
            start = time.perf_counter_ns()
            for _ in range(args.iterations):
                mx.eval(*fn(logits, axis))
            mx.synchronize()
            elapsed = time.perf_counter_ns() - start
            samples[name].append(elapsed / args.iterations / 1_000)

    print(f"MLX {mx.__version__}; device: {mx.default_device()}")
    print(f"Input: {logits.shape}, {logits.dtype}; seed: {args.seed}")
    print(f"All outputs match: index={reference[0].item()}, max={reference[1].item()}")
    print(f"Warmup: {args.warmup}; rounds: {args.rounds}; calls/round: {args.iterations}")
    print("Synchronized wall time per call (includes Python, dispatch, and GPU work).")
    print("Same resident input reused; median and range of round means, in microseconds.")
    baseline = statistics.median(samples["argmax + max"])
    print(f"{'Variant':32} {'Median us':>12} {'Min us':>12} {'Max us':>12} {'Speedup':>10}")
    for name, values in samples.items():
        median = statistics.median(values)
        print(f"{name:32} {median:12.3f} {min(values):12.3f} {max(values):12.3f} {baseline / median:9.2f}x")
    print("Speedup is relative to uncompiled argmax + max.")

    if args.capture:
        args.capture_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=args.capture_dir)).resolve()
        filenames = ("gather", "gather_compiled", "reduce", "reduce_compiled")
        for (name, fn), filename in zip(variants.items(), filenames):
            path = run_dir / f"{filename}.gputrace"
            capture(fn, logits, axis, path)
            print(f"Captured {name}: {path}")
        print("Captures contain one fresh call each; input generation and warmup are excluded.")


if __name__ == "__main__":
    main()

