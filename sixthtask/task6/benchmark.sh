#!/usr/bin/env bash
set -e

sizes="128 256 512 1024"
targets="host multicore gpu"
eps="1e-6"
iters="1000000"
csv="benchmark.csv"

echo "target,size,iterations,error,time_s" > "$csv"

for target in $targets; do
    build_dir="build-$target"
    cmake -S . -B "$build_dir" -DACC_TARGET="$target"
    cmake --build "$build_dir"

    for size in $sizes; do
        output=$("./$build_dir/task6" --size "$size" --eps "$eps" --iters "$iters")
        iterations=$(echo "$output" | awk -F= '/iterations/ {print $2}')
        error=$(echo "$output" | awk -F= '/error/ {print $2}')
        time_s=$(echo "$output" | awk -F= '/time/ {print $2}')
        echo "$target,$size,$iterations,$error,$time_s" >> "$csv"
    done
done

echo "Saved $csv"
