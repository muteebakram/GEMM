#!/bin/bash

OUTPUT=openmp.txt

function run() {
  /opt/homebrew/Cellar/llvm/17.0.4/bin/clang -O3 -fopenmp mm4_main.c mm4_par.c && ./a.out $1 $2 $3 | tee -a $OUTPUT
  printf "\n---------------------------------------------------------------------------------------\n"
}

>$OUTPUT

run 16 16 16
run 32 32 32
# run 1024 1024 1024
