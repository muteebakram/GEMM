#include <stdio.h>
#include <stdlib.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))

void transpose_non_multiple(float *__restrict__ Bt, const float *__restrict__ B, int Nk, int Nj)
{
  int j, k;
#pragma omp parallel for
  for (int k = 0; k < Nk; k++)
    for (j = 0; j < Nj; j++)
      Bt[k * Nj + j] = B[j * Nk + k];
}

void transpose_multiple(float *__restrict__ Bt, const float *__restrict__ B, int Nk, int Nj)
{
  int j, k;
#pragma omp parallel for
  for (int k = 0; k < Nk; k += 4)
  {
    int kNj = k * Nj, k1Nj = (k + 1) * Nj, k2Nj = (k + 2) * Nj, k3Nj = (k + 3) * Nj;
    for (j = 0; j < Nj; j += 4)
    {
      int jNk = j * Nk, j1Nk = (j + 1) * Nk, j2Nk = (j + 2) * Nk, j3Nk = (j + 3) * Nk;
      Bt[kNj + j] = B[jNk + k];
      Bt[k1Nj + j] = B[jNk + k + 1];
      Bt[k2Nj + j] = B[jNk + k + 2];
      Bt[k3Nj + j] = B[jNk + k + 3];

      Bt[kNj + j + 1] = B[j1Nk + k];
      Bt[k1Nj + j + 1] = B[j1Nk + k + 1];
      Bt[k2Nj + j + 1] = B[j1Nk + k + 2];
      Bt[k3Nj + j + 1] = B[j1Nk + k + 3];

      Bt[kNj + j + 2] = B[j2Nk + k];
      Bt[k1Nj + j + 2] = B[j2Nk + k + 1];
      Bt[k2Nj + j + 2] = B[j2Nk + k + 2];
      Bt[k3Nj + j + 2] = B[j2Nk + k + 3];

      Bt[kNj + j + 3] = B[j3Nk + k];
      Bt[k1Nj + j + 3] = B[j3Nk + k + 1];
      Bt[k2Nj + j + 3] = B[j3Nk + k + 2];
      Bt[k3Nj + j + 3] = B[j3Nk + k + 3];
    }
  }
}

void ab_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  if (Ni * Nj > 4 * Nk && Nk % 2 == 0)
  {
    // printf("1\n");
    // 1. Loop permutation to ikj; 2. Loop k unroll 2; 3. Use scalar variables.
#pragma omp parallel for schedule(dynamic) private(i, j, k)
    for (i = 0; i < Ni; i++)
    {
      int i_Nj = i * Nj;
      int i_Nk = i * Nk;

      for (k = 0; k < Nk; k += 2)
      {
        int k_Nj = k * Nj;
        const float *A_i_k = &A[i_Nk + k];

        for (j = 0; j < Nj; j++)
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
#pragma omp parallel for schedule(dynamic) private(i, j, k)
    for (i = 0; i < Ni; i++)
    {
      int i_Nj = i * Nj; // i * Nj
      int i_Nk = i * Nk; // i * Nk

      int rem = Nk % 9;
      for (k = 0; k < rem; k++)
        for (j = 0; j < Nj; j++)
          C[i * Nj + j] += A[i * Nk + k] * B[k * Nj + j];

      for (k = rem; k < Nk; k += 9)
      {
        int k_Nj = k * Nj; // k * Nj
        const float *A_i_k = &A[i_Nk + k];

        for (j = 0; j < Nj; j++)
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
    int kt = 0, TILE = 32;

    // 1. Loop permutation to ikj; 2. Tile Loop k by 32; 3. Loop k unroll by 8; 4. Use scalar variables.
    // TODO Still need improve when Nk >>> Ni * Nj
    // #pragma omp parallel for schedule(dynamic) private(i, j, k, kt)
    //     for (kt = 0; kt < Nk; kt += TILE)
    //       for (i = 0; i < Ni; i++)
    //         for (j = 0; j < Nj; j++)
    //           for (k = kt; k < kt + TILE; k++)
    //             C[i * Nj + j] += A[i * Nk + k] * B[k * Nj + j];

#pragma omp parallel for schedule(dynamic) private(i, j, k)
    for (i = 0; i < Ni; i++)
    {
      int i_Nj = i * Nj; // i * Nj
      int i_Nk = i * Nk; // i * Nk

      for (kt = 0; kt < Nk; kt += TILE)
        for (k = kt; k < kt + TILE; k += 8)
        {
          int k_Nj = k * Nj; // k * Nj
          const float *A_i_k = &A[i_Nk + k];

          for (j = 0; j < Nj; j++)
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
          }
        }
    }
  }
  else if (Ni * Nj <= 4 * Nk && Nk % 2 != 0)
  {
    // printf("4\n");
    // 1. Loop permutation to ikj; 2. Tile Loop k by 36; 3. Loop k unroll by 9; 4. Use scalar variables.
#pragma omp parallel for schedule(dynamic) private(i, j, k)
    for (i = 0; i < Ni; i++)
    {
      int i_Nj = i * Nj; // i * Nj
      int i_Nk = i * Nk; // i * Nk

      for (int kt = 0; kt < Nk; kt += 36)
        for (k = kt; k < min(kt + 36, Nk); k += 9)
        {
          int k_Nj = k * Nj; // k * Nj
          const float *A_i_k = &A[i_Nk + k];

          for (j = 0; j < Nj; j++)
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
#pragma omp parallel for schedule(dynamic) private(i, j, k)
    for (i = 0; i < Ni; i++)
      for (k = 0; k < Nk; k++)
        for (j = 0; j < Nj; j++)
          C[i * Nj + j] += A[i * Nk + k] * B[k * Nj + j];
  }
}

// TODO Add scalar varaible to reuse values.
void aTb_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k, kt;
  if ((Ni * Nj > 4 * Nk) && Nk % 2 == 0)
  {
    // printf("1\n");
#pragma omp parallel for schedule(dynamic) private(i, j, k)
    for (i = 0; i < Ni; i++)
    {
      for (k = 0; k < Nk; k += 2)
        for (j = 0; j < Nj; j++)
        {
          // C[i][j] += A[k][i]*B[k][j];
          C[i * Nj + j] += A[k * Ni + i] * B[k * Nj + j];
          C[i * Nj + j] += A[(k + 1) * Ni + i] * B[(k + 1) * Nj + j];
        }
    }
  }
  else if ((Ni * Nj > 4 * Nk) && Nk % 2 != 0)
  {
    // printf("2\n");
#pragma omp parallel for schedule(dynamic) private(i, j, k)
    for (i = 0; i < Ni; i++)
    {
      int rem = Nk % 3;
      for (k = 0; k < rem; k++)
        for (j = 0; j < Nj; j++)
        {
          // C[i][j] += A[k][i]*B[k][j];
          C[i * Nj + j] += A[k * Ni + i] * B[k * Nj + j];
        }
      for (k = rem; k < Nk; k += 3)
        for (j = 0; j < Nj; j++)
        {
          // C[i][j] += A[k][i]*B[k][j];
          C[i * Nj + j] += A[k * Ni + i] * B[k * Nj + j];
          C[i * Nj + j] += A[(k + 1) * Ni + i] * B[(k + 1) * Nj + j];
          C[i * Nj + j] += A[(k + 2) * Ni + i] * B[(k + 2) * Nj + j];
        }
    }
  }
  else if ((Ni * Nj <= 4 * Nk))
  {
    // printf("3\n");
#pragma omp parallel for schedule(dynamic) private(i, j, k, kt)
    for (i = 0; i < Ni; i++)
    {
      int TILE = 32;
      for (int kt = 0; kt < Nk; kt += TILE)
        for (k = kt; k < min(kt + TILE, Nk); k += 4)
          for (j = 0; j < Nj; j++)
          {
            // C[i][j] += A[k][i]*B[k][j];
            C[i * Nj + j] += A[k * Ni + i] * B[k * Nj + j];
            C[i * Nj + j] += A[(k + 1) * Ni + i] * B[(k + 1) * Nj + j];
            C[i * Nj + j] += A[(k + 2) * Ni + i] * B[(k + 2) * Nj + j];
            C[i * Nj + j] += A[(k + 3) * Ni + i] * B[(k + 3) * Nj + j];
            // C[i * Nj + j] += A[(k + 4) * Ni + i] * B[(k + 4) * Nj + j];
            // C[i * Nj + j] += A[(k + 5) * Ni + i] * B[(k + 5) * Nj + j];
            // C[i * Nj + j] += A[(k + 6) * Ni + i] * B[(k + 6) * Nj + j];
            // C[i * Nj + j] += A[(k + 7) * Ni + i] * B[(k + 7) * Nj + j];
            // C[i * Nj + j] += A[(k + 8) * Ni + i] * B[(k + 8) * Nj + j];
          }
    }
  }
  else
  {
    // printf("3\n");
    // With given test case we won't enter these case. Added as a safe holder.
#pragma omp parallel for schedule(dynamic) private(i, j, k)
    for (i = 0; i < Ni; i++)
      for (k = 0; k < Nk; k++)
        for (j = 0; j < Nj; j++)
          // C[i][j] += A[k][i]*B[k][j];
          C[i * Nj + j] += A[k * Ni + i] * B[k * Nj + j];
  }
}

void abT_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, it, j, jt, k;
  if ((Ni * Nj > 4 * Nk) && Nk % 2 == 0)
  {
    // printf("1\n");
    float sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15, sum16;
#pragma omp parallel for schedule(dynamic) private(i, j, k) reduction(+ : sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15, sum16)
    for (i = 0; i < Ni; i += 4)
      for (j = 0; j < Nj; j += 4)
      {
        sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0, sum8 = 0, sum9 = 0, sum10 = 0, sum11 = 0, sum12 = 0, sum13 = 0, sum14 = 0, sum15 = 0, sum16 = 0;
        for (k = 0; k < Nk; k++)
        {
          // C[i][j] += A[i][k]*B[j][k];
          sum1 += A[i * Nk + k] * B[j * Nk + k];
          sum2 += A[(i + 1) * Nk + k] * B[j * Nk + k];
          sum3 += A[(i + 2) * Nk + k] * B[j * Nk + k];
          sum4 += A[(i + 3) * Nk + k] * B[j * Nk + k];

          sum5 += A[i * Nk + k] * B[(j + 1) * Nk + k];
          sum6 += A[(i + 1) * Nk + k] * B[(j + 1) * Nk + k];
          sum7 += A[(i + 2) * Nk + k] * B[(j + 1) * Nk + k];
          sum8 += A[(i + 3) * Nk + k] * B[(j + 1) * Nk + k];

          sum9 += A[i * Nk + k] * B[(j + 2) * Nk + k];
          sum10 += A[(i + 1) * Nk + k] * B[(j + 2) * Nk + k];
          sum11 += A[(i + 2) * Nk + k] * B[(j + 2) * Nk + k];
          sum12 += A[(i + 3) * Nk + k] * B[(j + 2) * Nk + k];

          sum13 += A[i * Nk + k] * B[(j + 3) * Nk + k];
          sum14 += A[(i + 1) * Nk + k] * B[(j + 3) * Nk + k];
          sum15 += A[(i + 2) * Nk + k] * B[(j + 3) * Nk + k];
          sum16 += A[(i + 3) * Nk + k] * B[(j + 3) * Nk + k];
        }
        C[i * Nj + j] = sum1;
        C[(i + 1) * Nj + j] = sum2;
        C[(i + 2) * Nj + j] = sum3;
        C[(i + 3) * Nj + j] = sum4;

        C[i * Nj + j + 1] = sum5;
        C[(i + 1) * Nj + j + 1] = sum6;
        C[(i + 2) * Nj + j + 1] = sum7;
        C[(i + 3) * Nj + j + 1] = sum8;

        C[i * Nj + j + 2] = sum9;
        C[(i + 1) * Nj + j + 2] = sum10;
        C[(i + 2) * Nj + j + 2] = sum11;
        C[(i + 3) * Nj + j + 2] = sum12;

        C[i * Nj + j + 3] = sum13;
        C[(i + 1) * Nj + j + 3] = sum14;
        C[(i + 2) * Nj + j + 3] = sum15;
        C[(i + 3) * Nj + j + 3] = sum16;
      }
  }
  else if ((Ni * Nj > 4 * Nk) && Nk % 2 != 0)
  {
    // printf("2\n");
    float sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15, sum16;

    int rem_i = Ni % 3;
#pragma omp parallel for schedule(dynamic) private(i, j, k)
    for (i = 0; i < rem_i; i++)
      for (j = 0; j < Nj; j++)
        for (k = 0; k < Nk; k++)
          C[i * Nj + j] = A[i * Nk + k] * B[j * Nk + k];

#pragma omp parallel for schedule(dynamic) private(i, j, k) reduction(+ : sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15, sum16)
    for (i = rem_i; i < Ni; i += 3)
    {
      int rem_j = Nj % 3;
      for (j = 0; j < rem_j; j++)
        for (k = 0; k < Nk; k++)
          C[i * Nj + j] = A[i * Nk + k] * B[j * Nk + k];

      for (j = rem_j; j < Nj; j += 3)
      {
        sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0, sum8 = 0, sum9 = 0, sum10 = 0, sum11 = 0, sum12 = 0, sum13 = 0, sum14 = 0, sum15 = 0, sum16 = 0;
        for (k = 0; k < Nk; k++)
        {
          // C[i][j] += A[i][k]*B[j][k];
          sum1 += A[i * Nk + k] * B[j * Nk + k];
          sum2 += A[(i + 1) * Nk + k] * B[j * Nk + k];
          sum3 += A[(i + 2) * Nk + k] * B[j * Nk + k];
          // sum4 += A[(i + 3) * Nk + k] * B[j * Nk + k];

          sum5 += A[i * Nk + k] * B[(j + 1) * Nk + k];
          sum6 += A[(i + 1) * Nk + k] * B[(j + 1) * Nk + k];
          sum7 += A[(i + 2) * Nk + k] * B[(j + 1) * Nk + k];
          // sum8 += A[(i + 3) * Nk + k] * B[(j + 1) * Nk + k];

          sum9 += A[i * Nk + k] * B[(j + 2) * Nk + k];
          sum10 += A[(i + 1) * Nk + k] * B[(j + 2) * Nk + k];
          sum11 += A[(i + 2) * Nk + k] * B[(j + 2) * Nk + k];
          // sum12 += A[(i + 3) * Nk + k] * B[(j + 2) * Nk + k];

          // sum13 += A[i * Nk + k] * B[(j + 3) * Nk + k];
          // sum14 += A[(i + 1) * Nk + k] * B[(j + 3) * Nk + k];
          // sum15 += A[(i + 2) * Nk + k] * B[(j + 3) * Nk + k];
          // sum16 += A[(i + 3) * Nk + k] * B[(j + 3) * Nk + k];
        }
        C[i * Nj + j] = sum1;
        C[(i + 1) * Nj + j] = sum2;
        C[(i + 2) * Nj + j] = sum3;
        // C[(i + 3) * Nj + j] = sum4;

        C[i * Nj + j + 1] = sum5;
        C[(i + 1) * Nj + j + 1] = sum6;
        C[(i + 2) * Nj + j + 1] = sum7;
        // C[(i + 3) * Nj + j + 1] = sum8;

        C[i * Nj + j + 2] = sum9;
        C[(i + 1) * Nj + j + 2] = sum10;
        C[(i + 2) * Nj + j + 2] = sum11;
        // C[(i + 3) * Nj + j + 2] = sum12;

        // C[i * Nj + j + 3] = sum13;
        // C[(i + 1) * Nj + j + 3] = sum14;
        // C[(i + 2) * Nj + j + 3] = sum15;
        // C[(i + 3) * Nj + j + 3] = sum16;
      }
    }
    //     #pragma omp parallel for schedule(dynamic) private(i, j, k) reduction(+ : sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15, sum16)
    // for (i = rem_i; i < Ni; i += 3)
    // {
    //   // int rem_j = Nj % 3;
    //   // for (j = 0; j < rem_j; j++)
    //   //   for (k = 0; k < Nk; k++)
    //   //     C[i * Nj + j] = A[i * Nk + k] * B[j * Nk + k];

    //   for (j = 0; j < Nj; j++)
    //   {
    //     sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0, sum8 = 0, sum9 = 0, sum10 = 0, sum11 = 0, sum12 = 0, sum13 = 0, sum14 = 0, sum15 = 0, sum16 = 0;
    //     for (k = 0; k < Nk; k++)
    //     {
    //       // C[i][j] += A[i][k]*B[j][k];
    //       sum1 += A[i * Nk + k] * B[j * Nk + k];
    //       sum2 += A[(i + 1) * Nk + k] * B[j * Nk + k];
    //       sum3 += A[(i + 2) * Nk + k] * B[j * Nk + k];
    //       sum4 += A[(i + 3) * Nk + k] * B[j * Nk + k];
    //       sum5 += A[(i + 4) * Nk + k] * B[j * Nk + k];
    //       sum6 += A[(i + 5) * Nk + k] * B[j * Nk + k];
    //       sum7 += A[(i + 6) * Nk + k] * B[j * Nk + k];
    //       sum8 += A[(i + 7) * Nk + k] * B[j * Nk + k];
    //       sum9 += A[(i + 8) * Nk + k] * B[j * Nk + k];

    //       // sum5 += A[i * Nk + k] * B[(j + 1) * Nk + k];
    //       // sum6 += A[(i + 1) * Nk + k] * B[(j + 1) * Nk + k];
    //       // sum7 += A[(i + 2) * Nk + k] * B[(j + 1) * Nk + k];
    //       // sum8 += A[(i + 3) * Nk + k] * B[(j + 1) * Nk + k];

    //       // sum9 += A[i * Nk + k] * B[(j + 2) * Nk + k];
    //       // sum10 += A[(i + 1) * Nk + k] * B[(j + 2) * Nk + k];
    //       // sum11 += A[(i + 2) * Nk + k] * B[(j + 2) * Nk + k];
    //       // sum12 += A[(i + 3) * Nk + k] * B[(j + 2) * Nk + k];

    //       // sum13 += A[i * Nk + k] * B[(j + 3) * Nk + k];
    //       // sum14 += A[(i + 1) * Nk + k] * B[(j + 3) * Nk + k];
    //       // sum15 += A[(i + 2) * Nk + k] * B[(j + 3) * Nk + k];
    //       // sum16 += A[(i + 3) * Nk + k] * B[(j + 3) * Nk + k];
    //     }
    //     C[i * Nj + j] = sum1;
    //     C[(i + 1) * Nj + j] = sum2;
    //     C[(i + 2) * Nj + j] = sum3;
    //     C[(i + 3) * Nj + j] = sum4;
    //     C[(i + 4) * Nj + j] = sum5;
    //     C[(i + 5) * Nj + j] = sum6;
    //     C[(i + 6) * Nj + j] = sum7;
    //     C[(i + 7) * Nj + j] = sum8;
    //     C[(i + 8) * Nj + j] = sum9;

    //     // C[i * Nj + j + 1] = sum5;
    //     // C[(i + 1) * Nj + j + 1] = sum6;
    //     // C[(i + 2) * Nj + j + 1] = sum7;
    //     // // C[(i + 3) * Nj + j + 1] = sum8;

    //     // C[i * Nj + j + 2] = sum9;
    //     // C[(i + 1) * Nj + j + 2] = sum10;
    //     // C[(i + 2) * Nj + j + 2] = sum11;
    //     // C[(i + 3) * Nj + j + 2] = sum12;

    //     // C[i * Nj + j + 3] = sum13;
    //     // C[(i + 1) * Nj + j + 3] = sum14;
    //     // C[(i + 2) * Nj + j + 3] = sum15;
    //     // C[(i + 3) * Nj + j + 3] = sum16;
    //   }
    // }
  }
  // QUESTION : how to parallelize k loop with reduction
  else if ((Ni * Nj <= 4 * Nk) && (Nk % 2 == 0)) // TODO Implement 3 and 4
  {
    // printf("3\n");

    int kt = 0, TILE = 32;
    float *Bt = (float *)malloc(sizeof(float) * Nk * Nj);
    transpose_multiple(Bt, B, Nk, Nj);

#pragma omp parallel for
    for (i = 0; i < Ni; i++)
    {
      int i_Nj = i * Nj; // i * Nj
      int i_Nk = i * Nk; // i * Nk

      for (kt = 0; kt < Nk; kt += TILE)
        for (k = kt; k < kt + TILE; k += 8)
        {
          int k_Nj = k * Nj; // k * Nj
          const float *A_i_k = &A[i_Nk + k];

          for (j = 0; j < Nj; j++)
          {
            int i_Nj_j = i_Nj + j; // i * Nj + j
            int k_Nj_j = k_Nj + j; // k * Nj + j

            C[i_Nj_j] += *A_i_k * Bt[k_Nj_j]; // A[i * Nk + k] * B[k * Nj + j]
            C[i_Nj_j] += *(A_i_k + 1) * Bt[k_Nj_j + (1 * Nj)];
            C[i_Nj_j] += *(A_i_k + 2) * Bt[k_Nj_j + (2 * Nj)];
            C[i_Nj_j] += *(A_i_k + 3) * Bt[k_Nj_j + (3 * Nj)];
            C[i_Nj_j] += *(A_i_k + 4) * Bt[k_Nj_j + (4 * Nj)];
            C[i_Nj_j] += *(A_i_k + 5) * Bt[k_Nj_j + (5 * Nj)];
            C[i_Nj_j] += *(A_i_k + 6) * Bt[k_Nj_j + (6 * Nj)];
            C[i_Nj_j] += *(A_i_k + 7) * Bt[k_Nj_j + (7 * Nj)];
          }
        }
    }
  }
  else if ((Ni * Nj <= 4 * Nk) && (Nk % 2 != 0))
  {
    // printf("4\n");
    int rem = Nk % 3;
    float *Bt = (float *)malloc(sizeof(float) * Nk * Nj);
    transpose_non_multiple(Bt, B, Nk, Nj);

#pragma omp parallel for
    for (i = 0; i < Ni; i++)
    {
      int i_Nj = i * Nj; // i * Nj
      int i_Nk = i * Nk; // i * Nk

      for (k = 0; k < rem; k++)
        for (j = 0; j < Nj; j++)
          C[i_Nj + j] += A[i_Nk + k] * Bt[k * Nj + j];

      for (k = rem; k < Nk; k += 3)
        for (j = 0; j < Nj; j++)
        {
          // C[i][j] += A[k][i]*B[k][j];
          C[i_Nj + j] += A[i_Nk + k] * Bt[k * Nj + j];
          C[i_Nj + j] += A[i_Nk + k + 1] * Bt[(k + 1) * Nj + j];
          C[i_Nj + j] += A[i_Nk + k + 2] * Bt[(k + 2) * Nj + j];
        }
    }
  }
  else
  {
// printf("5\n");
#pragma omp parallel for
    for (i = 0; i < Ni; i++)
      for (j = 0; j < Nj; j++)
        for (k = 0; k < Nk; k++)
          C[i * Nj + j] += A[i * Nk + k] * B[j * Nj + k];
  }
}

void aTbT_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, jt, k;
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
