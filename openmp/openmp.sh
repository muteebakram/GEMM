#!/bin/bash

OUTPUT=AtB_result.txt
GCC=gcc
# GCC=/opt/homebrew/Cellar/llvm/17.0.4/bin/clang # Macbook Apple Silicon

>$OUTPUT

function run() {
  # $GCC -O3 -fopenmp mm4_main.c mm4_par.c && time ./a.out $1 $2 $3
  $GCC -O3 -fopenmp mm4_main.c mm4_par.c && time ./a.out $1 $2 $3 | tee -a $OUTPUT
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
