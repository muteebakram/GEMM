#include <stdio.h>
#include <stdlib.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))

void transpose_non_multiple(float *__restrict__ Bt, const float *__restrict__ B, int Nk, int Nj)
{
#pragma omp parallel for
  for (int k = 0; k < Nk; k++)
    for (int j = 0; j < Nj; j++)
      Bt[k * Nj + j] = B[j * Nk + k];
}

void transpose_multiple(float *__restrict__ Bt, const float *__restrict__ B, int Nk, int Nj)
{
#pragma omp parallel for
  for (int k = 0; k < Nk; k += 4)
  {
    int kNj = k * Nj, k1Nj = (k + 1) * Nj, k2Nj = (k + 2) * Nj, k3Nj = (k + 3) * Nj;
    for (int j = 0; j < Nj; j += 4)
    {
      int jNk_k = j * Nk + k, j1Nk_k = (j + 1) * Nk + k, j2Nk_k = (j + 2) * Nk + k, j3Nk_k = (j + 3) * Nk + k;

      Bt[kNj + j] = B[jNk_k];
      Bt[k1Nj + j] = B[jNk_k + 1];
      Bt[k2Nj + j] = B[jNk_k + 2];
      Bt[k3Nj + j] = B[jNk_k + 3];

      Bt[kNj + j + 1] = B[j1Nk_k];
      Bt[k1Nj + j + 1] = B[j1Nk_k + 1];
      Bt[k2Nj + j + 1] = B[j1Nk_k + 2];
      Bt[k3Nj + j + 1] = B[j1Nk_k + 3];

      Bt[kNj + j + 2] = B[j2Nk_k];
      Bt[k1Nj + j + 2] = B[j2Nk_k + 1];
      Bt[k2Nj + j + 2] = B[j2Nk_k + 2];
      Bt[k3Nj + j + 2] = B[j2Nk_k + 3];

      Bt[kNj + j + 3] = B[j3Nk_k];
      Bt[k1Nj + j + 3] = B[j3Nk_k + 1];
      Bt[k2Nj + j + 3] = B[j3Nk_k + 2];
      Bt[k3Nj + j + 3] = B[j3Nk_k + 3];
    }
  }
}

void ab_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  if (Ni * Nj > 4 * Nk && Nk % 2 == 0)
  {
    // printf("1\n");
    // 1. Loop permutation to ikj; 2. Loop k unroll 2; 3. Use scalar variables.
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < Ni; i++)
    {
      int i_Nj = i * Nj;
      int i_Nk = i * Nk;

      for (int k = 0; k < Nk; k += 2)
      {
        int k_Nj = k * Nj;
        const float *A_i_k = &A[i_Nk + k];

        for (int j = 0; j < Nj; j++)
        {
          C[i_Nj + j] += *A_i_k * B[k_Nj + j];
          C[i_Nj + j] += *(A_i_k + 1) * B[k_Nj + Nj + j];
        }
      }
    }
  }
  else if (Ni * Nj > 4 * Nk && Nk % 2 != 0)
  {
    // printf("2\n");
    // 1. Loop permutation to ikj; 2. Loop k unroll by 9 with remainder; 3. Use scalar variables.
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < Ni; i++)
    {
      int i_Nj = i * Nj; // i * Nj
      int i_Nk = i * Nk; // i * Nk

      int rem = Nk % 9;
      for (int k = 0; k < rem; k++)
        for (int j = 0; j < Nj; j++)
          C[i * Nj + j] += A[i * Nk + k] * B[k * Nj + j];

      for (int k = rem; k < Nk; k += 9)
      {
        int k_Nj = k * Nj; // k * Nj
        const float *A_i_k = &A[i_Nk + k];

        for (int j = 0; j < Nj; j++)
        {
          int i_Nj_j = i_Nj + j; // i * Nj + j
          int k_Nj_j = k_Nj + j; // k * Nj + j

          C[i_Nj_j] += *A_i_k * B[k_Nj_j]; // A[i * Nk + k] * B[k * Nj + j]
          C[i_Nj_j] += *(A_i_k + 1) * B[k_Nj_j + (1 * Nj)];
          C[i_Nj_j] += *(A_i_k + 2) * B[k_Nj_j + (2 * Nj)];
          C[i_Nj_j] += *(A_i_k + 3) * B[k_Nj_j + (3 * Nj)];
          C[i_Nj_j] += *(A_i_k + 4) * B[k_Nj_j + (4 * Nj)];
          C[i_Nj_j] += *(A_i_k + 5) * B[k_Nj_j + (5 * Nj)];
          C[i_Nj_j] += *(A_i_k + 6) * B[k_Nj_j + (6 * Nj)];
          C[i_Nj_j] += *(A_i_k + 7) * B[k_Nj_j + (7 * Nj)];
          C[i_Nj_j] += *(A_i_k + 8) * B[k_Nj_j + (8 * Nj)];
        }
      }
    }
  }

  else if (Ni * Nj <= 4 * Nk && Nk % 2 == 0)
  {
    // printf("3\n");
    // int kt = 0, TILE = 32;

#pragma omp parallel for
    for (int i = 0; i < Ni; i++)
    {
      int i_Nj = i * Nj; // i * Nj
      int i_Nk = i * Nk; // i * Nk

      for (int k = 0; k < Nk; k += 8)
      // for (int k = kt; k < min(kt + TILE, Nk); k += 9)
      {
        int k_Nj = k * Nj; // k * Nj

        for (int j = 0; j < Nj; j++)
        {
          int i_Nj_j = i_Nj + j; // i * Nj + j
          int k_Nj_j = k_Nj + j; // k * Nj + j

          float *C_i_j = &C[i_Nj + j];
          const float *B_k_j = &B[k_Nj + j];

          *C_i_j += A[i_Nk + k] * *(B_k_j); // A[i * Nk + k] * B[k * Nj + j]
          *C_i_j += A[i_Nk + k + 1] * *(B_k_j + 1 * Nj);
          *C_i_j += A[i_Nk + k + 2] * *(B_k_j + 2 * Nj);
          *C_i_j += A[i_Nk + k + 3] * *(B_k_j + 3 * Nj);
          *C_i_j += A[i_Nk + k + 4] * *(B_k_j + 4 * Nj);
          *C_i_j += A[i_Nk + k + 5] * *(B_k_j + 5 * Nj);
          *C_i_j += A[i_Nk + k + 6] * *(B_k_j + 6 * Nj);
          *C_i_j += A[i_Nk + k + 7] * *(B_k_j + 7 * Nj);
        }
      }
    }
  }
  else if (Ni * Nj <= 4 * Nk && Nk % 2 != 0)
  {
    // printf("4\n");
    // 1. Loop permutation to ikj; 2. Tile Loop k by 36; 3. Loop k unroll by 9; 4. Use scalar variables.
    int TILE = 36;

#pragma omp parallel for
    for (int i = 0; i < Ni; i++)
    {
      int i_Nj = i * Nj; // i * Nj
      int i_Nk = i * Nk; // i * Nk

      for (int kt = 0; kt < Nk; kt += TILE)
        for (int k = kt; k < min(kt + TILE, Nk); k += 9)
        {
          int k_Nj = k * Nj; // k * Nj
          const float *A_i_k = &A[i_Nk + k];

          for (int j = 0; j < Nj; j++)
          {
            int i_Nj_j = i_Nj + j; // i * Nj + j
            int k_Nj_j = k_Nj + j; // k * Nj + j

            C[i_Nj_j] += *A_i_k * B[k_Nj_j]; // A[i * Nk + k] * B[k * Nj + j]
            C[i_Nj_j] += *(A_i_k + 1) * B[k_Nj_j + (1 * Nj)];
            C[i_Nj_j] += *(A_i_k + 2) * B[k_Nj_j + (2 * Nj)];
            C[i_Nj_j] += *(A_i_k + 3) * B[k_Nj_j + (3 * Nj)];
            C[i_Nj_j] += *(A_i_k + 4) * B[k_Nj_j + (4 * Nj)];
            C[i_Nj_j] += *(A_i_k + 5) * B[k_Nj_j + (5 * Nj)];
            C[i_Nj_j] += *(A_i_k + 6) * B[k_Nj_j + (6 * Nj)];
            C[i_Nj_j] += *(A_i_k + 7) * B[k_Nj_j + (7 * Nj)];
            C[i_Nj_j] += *(A_i_k + 8) * B[k_Nj_j + (8 * Nj)];
          }
        }
    }
  }
  else
  {
    // printf("5\n");
    // With given test case we won't enter these case. Added as a safe holder.
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < Ni; i++)
      for (int k = 0; k < Nk; k++)
        for (int j = 0; j < Nj; j++)
          C[i * Nj + j] += A[i * Nk + k] * B[k * Nj + j];
  }
}

void aTb_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  float *At = (float *)malloc(sizeof(float) * Nk * Nj);

  if (Nk % 2 == 0)
    transpose_multiple(At, A, Ni, Nk);
  else
    transpose_non_multiple(At, A, Ni, Nk);

  ab_par(At, B, C, Ni, Nj, Nk);
}

void abT_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  float *Bt = (float *)malloc(sizeof(float) * Nk * Nj);

  if (Nk % 2 == 0)
    transpose_multiple(Bt, B, Nk, Nj);
  else
    transpose_non_multiple(Bt, B, Nk, Nj);

  ab_par(A, Bt, C, Ni, Nj, Nk);
}

void aTbT_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  float *At = (float *)malloc(sizeof(float) * Nk * Nj);
  float *Bt = (float *)malloc(sizeof(float) * Nk * Nj);

  if (Nk % 2 == 0)
  {
    transpose_multiple(At, A, Ni, Nk);
    transpose_multiple(Bt, B, Nk, Nj);
  }
  else
  {
    transpose_non_multiple(At, A, Ni, Nk);
    transpose_non_multiple(Bt, B, Nk, Nj);
  }

  ab_par(At, Bt, C, Ni, Nj, Nk);
}
