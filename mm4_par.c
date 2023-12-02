/*
1. First optimization - loop permutation
2. But loop permutation with inner loop size more where to put openmp pragma?
3.
*/

// 1. Perform loop transform form ijk to ikj.
// 2. Unroll i loop by 2
// 3. Unroll k loop by 2
// 4. Use scalar values for A matrix.
void ab_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  if (i % 2 == 0 && k % 2 == 0)
  {
#pragma omp parallel for schedule(dynamic) private(i, j, k)
    for (i = 0; i < Ni; i += 2)
    {
      for (k = 0; k < Nk; k += 2)
      {
        int ik = i * Nk + k;
        float Aik = A[ik];
        float Ai1k = A[ik + Nk];

        float Aik1 = A[ik + 1];
        float Ai1k1 = A[ik + Nk + 1];

        for (j = 0; j < Nj; j++)
        {
          // C[i][j] = C[i][j] + A[i][k]*B[k][j];
          int ij = i * Nj + j;
          int kj = k * Nj + j;

          C[ij] += Aik * B[kj];
          C[ij] += Aik1 * B[kj + Nj];

          C[ij + Nj] += Ai1k * B[kj];
          C[ij + Nj] += Ai1k1 * B[kj + Nj];
        }
      }
    }
  }

  // When k is very large it has to be outer loop and parallelized?
  // int i, j, k;
  // for (k = 0; k < Nk; k++)
  //   for (j = 0; j < Nj; j++)
  //     for (i = 0; i < Ni; i++)
  //       // C[i][j] = C[i][j] + A[i][k]*B[k][j];
  //       C[i * Nj + j] = C[i * Nj + j] + A[i * Nk + k] * B[k * Nj + j];
}

void aTb_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;
#pragma omp parallel for schedule(dynamic) private(i, j, k)
  for (i = 0; i < Ni; i++)
    for (k = 0; k < Nk; k++)
      for (j = 0; j < Nj; j++)
        // C[i][j] = C[i][j] + A[k][i]*B[k][j];
        C[i * Nj + j] += A[k * Ni + i] * B[k * Nj + j];
}

void abT_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;
#pragma omp parallel for schedule(dynamic) private(i, j, k)
  for (i = 0; i < Ni; i++)
    for (j = 0; j < Nj; j++)
      for (k = 0; k < Nk; k++)
        // C[i][j] = C[i][j] + A[i][k]*B[j][k];
        C[i * Nj + j] += A[i * Nk + k] * B[j * Nk + k];
}

void aTbT_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;
#pragma omp parallel for schedule(dynamic) private(i, j, k)
  for (i = 0; i < Ni; i++)
    for (j = 0; j < Nj; j++)
      for (k = 0; k < Nk; k++)
        // C[i][j] = C[i][j] + A[k][i]*B[j][k];
        C[i * Nj + j] += A[k * Ni + i] * B[j * Nk + k];
}
