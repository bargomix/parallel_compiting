#!/usr/bin/env bash
set -e

lock_dir=".task6.lock"
sizes="128 256 512 1024"
targets="host multicore gpu"
eps="1e-6"
iters="1000000"
csv="benchmark.csv"
profile_size="512"
profile_iters="100"
profile_dir="profiles"

if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "task6 benchmark/profile is already running"
    exit 1
fi
trap 'rmdir "$lock_dir"' EXIT

echo "target,size,iterations,error,time_s" > "$csv"

for target in $targets; do
    build_dir="build-$target"
    cmake -S . -B "$build_dir" -DACC_TARGET="$target"
    cmake --build "$build_dir"

    for size in $sizes; do
        echo "Run target=$target size=$size"
        output=$("./$build_dir/task6" --size "$size" --eps "$eps" --iters "$iters")
        iterations=$(echo "$output" | awk -F= '/iterations/ {print $2}')
        error=$(echo "$output" | awk -F= '/error/ {print $2}')
        time_s=$(echo "$output" | awk -F= '/time/ {print $2}')
        echo "$target,$size,$iterations,$error,$time_s" >> "$csv"
    done
done

echo "Saved $csv"

if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys not found, skip profiling"
    exit 0
fi

mkdir -p "$profile_dir"

for target in $targets; do
    build_dir="build-$target"
    report="$profile_dir/task6_${target}_${profile_size}_${profile_iters}"

    echo "Profile target=$target size=$profile_size iters=$profile_iters"
    nsys profile -f true -o "$report" "./$build_dir/task6" \
        --size "$profile_size" --eps "$eps" --iters "$profile_iters"
done

echo "Saved profiles to $profile_dir"
