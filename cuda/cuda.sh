#!/bin/bash

GCC=nvcc

OUTPUT=AB_result_new.txt

# mv $OUTPUT ABt_result_prev.txt # Compare previous run with new run.
>$OUTPUT # Clear the old contents.

function run() {
  # $GCC -O3 mm4_main.cu mm4_gpu.cu && time ./a.out $1 $2 $3
  $GCC -O3 mm4_main.cu mm4_gpu.cu && time ./a.out $1 $2 $3 | tee -a $OUTPUT
}

# Multiple sizes
run $((8 * 1024)) $((8 * 1024)) 16
run 4096 4096 64
run 2048 2048 256
run 1024 1024 1024
run 256 256 $((16 * 1024))
run 64 64 $((256 * 1024))
run 16 16 $((4 * 1024 * 1024))

# Non multiple sizes
run $((9 * 999)) $((9 * 999)) 37
run $((3 * 999)) $((3 * 999)) 111
run 999 999 999
run 333 333 $((9 * 999))
run 111 111 $((81 * 999))
run 37 37 $((81 * 999))