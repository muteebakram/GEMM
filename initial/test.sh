#!/bin/bash

OUTPUT=openmp.txt

function run() {
  echo "Running Ni $1, Nj $2, Nk $3..." | tee -a $OUTPUT
  gcc openmp.c && ./a.out $1 $2 $3 | tee -a $OUTPUT
}

>$OUTPUT

run 16 16 16
run $((8 * 1024)) $((8 * 1024)) 16
run 4096 4096 64
