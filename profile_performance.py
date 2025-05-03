#!/usr/bin/env python3
"""
Wind & Temperature Tile Server - Performance Testing
Tool to benchmark and compare performance of different parallelization strategies.
"""

import os
import time
import subprocess
import argparse
from datetime import datetime

def run_test(name, command, repeat=1, save_output=True):
    """Run a test and measure its performance."""
    print(f"\n===== Running test: {name} =====")
    print(f"Command: {command}")

    total_time = 0
    outputs = []

    for i in range(repeat):
        print(f"\nRun {i+1}/{repeat}")
        start_time = time.time()

        # Run the command and capture output
        result = subprocess.run(command, shell=True, text=True, capture_output=True)

        elapsed = time.time() - start_time
        total_time += elapsed
        outputs.append((result.stdout, result.stderr))

        # Print the output from this run
        print(result.stdout)
        if result.stderr:
            print(f"Errors:\n{result.stderr}")
        print(f"Run {i+1} time: {elapsed:.2f} seconds")

    avg_time = total_time / repeat
    print(f"\nTest '{name}' completed {repeat} runs")
    print(f"Average time: {avg_time:.2f} seconds")

    if save_output:
        # Save detailed results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"profile_results_{name.replace(' ', '_').lower()}_{timestamp}.txt"
        with open(filename, "w") as f:
            f.write(f"Test: {name}\n")
            f.write(f"Command: {command}\n")
            f.write(f"Average time over {repeat} runs: {avg_time:.2f} seconds\n\n")

            for i, (stdout, stderr) in enumerate(outputs):
                f.write(f"== Run {i+1} ==\n")
                f.write(stdout)
                if stderr:
                    f.write("\nErrors:\n")
                    f.write(stderr)
                f.write("\n\n")

        print(f"Results saved to {filename}")

    return avg_time

def main():
    parser = argparse.ArgumentParser(description="Test performance of different tile generation strategies")
    parser.add_argument("--zoom", type=str, default="0,1,2,3",
                        help="Zoom levels to use for testing (default: 0,1,2,3)")
    parser.add_argument("--threads", type=str, default="1,2,4,8",
                        help="Thread counts to test (comma-separated, default: 1,2,4,8)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of times to repeat each test (default: 1)")
    parser.add_argument("--output", type=str, default="profile_results.txt",
                        help="Output file for summary results (default: profile_results.txt)")
    args = parser.parse_args()

    # Parse the thread counts
    thread_counts = [int(t) for t in args.threads.split(",")]

    # Tests to run
    tests = [
        # ThreadPoolExecutor tests (original)
        *[{
            "name": f"ThreadPoolExecutor ({threads} threads)",
            "command": f"python grib_to_tiles.py --threads {threads} --zoom {args.zoom}"
        } for threads in thread_counts],

        # ProcessPoolExecutor tests (new)
        *[{
            "name": f"ProcessPoolExecutor ({threads} processes)",
            "command": f"python grib_to_tiles.py --threads {threads} --processes {threads} --zoom {args.zoom} --use-processes"
        } for threads in thread_counts],
    ]

    # Run each test and collect results
    results = []
    for test in tests:
        avg_time = run_test(test["name"], test["command"], repeat=args.repeat)
        results.append((test["name"], avg_time))

    # Print and save summary
    print("\n===== Performance Summary =====")

    # Find the baseline (slowest) time for comparison
    baseline = max(results, key=lambda x: x[1])[1]

    with open(args.output, "w") as f:
        f.write(f"Wind & Temperature Tile Server - Performance Test Results\n")
        f.write(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Zoom levels: {args.zoom}\n")
        f.write(f"Repetitions per test: {args.repeat}\n\n")

        f.write("Test configuration | Time (seconds) | Improvement\n")
        f.write("--------------------|----------------|------------\n")

        # Sort by performance (fastest first)
        for name, time in sorted(results, key=lambda x: x[1]):
            improvement = baseline / time
            print(f"{name}: {time:.2f}s ({improvement:.2f}x faster than baseline)")
            f.write(f"{name} | {time:.2f} | {improvement:.2f}x\n")

    print(f"\nDetailed summary saved to {args.output}")

if __name__ == "__main__":
    main()