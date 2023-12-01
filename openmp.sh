#!/bin/bash

OUTPUT=openmp.txt

function run() {
  echo "Running.."
  /opt/homebrew/Cellar/llvm/17.0.4/bin/clang -O3 -fopenmp mm4_main.c mm4_par.c && ./a.out | tee -a $OUTPUT
}

>$OUTPUT

run
