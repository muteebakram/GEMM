#include <stdio.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))

void ab_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  if (Ni * Nj > 4 * Nk)
  {
    if (Nk % 2 == 0)
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
    else
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
  }
  else
  {
    if (Nk % 2 == 0)
    {
      // printf("3\n");
      // 1. Loop permutation to ikj; 2. Tile Loop k by 32; 3. Loop k unroll by 8; 4. Use scalar variables.
      // TODO Still need improve when Nk >>> Ni * Nj
#pragma omp parallel for schedule(dynamic) private(i, j, k)
      for (i = 0; i < Ni; i++)
      {
        int i_Nj = i * Nj; // i * Nj
        int i_Nk = i * Nk; // i * Nk

        int TILE = 32;
        for (int kt = 0; kt < Nk; kt += TILE)
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
    else
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
  else if ((Ni * Nj <= 4 * Nk) && (Nk % 2 == 0)) // TODO Implement 3 and 4
  {
    printf("3\n");
    // float sum1;
//     for (i = 0; i < Ni; i++)
//       for (j = 0; j < Nj; j++)
//       {
//         sum1 = 0;
// // #pragma omp parallel for schedule(dynamic) firstprivate(i, j)
// // #pragma omp simd
// #pragma omp parallel for schedule(dynamic)  reduction(+ : sum1)
//         for (k = 0; k < Nk; k++)
//         {
//           // C[i][j] = C[i][j] + A[i][k]*B[j][k];
//           sum1 += A[i * Nk + k] * B[j * Nk + k];
//         }
//         C[i * Nj + j] = sum1;
//       }
  }
  //   int TILE = 32;

    // float sum1;

  // #pragma omp parallel for schedule(dynamic) private(it, jt) reduction(+ : sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)
  //   for (it = 0; it < Ni; it += TILE)
  //     for (jt = 0; jt < Nj; jt += TILE)
  //       for (i = it; i < it + TILE; i++)
  //         for (j = jt; j < jt + TILE; j++)
  //         {
  //           sum1 = 0;
  //           // sum2 = 0;
  //           // sum3 = 0;
  //           // sum4 = 0;
  //           // sum5 = 0;
  //           // sum6 = 0;
  //           for (k = 0; k < Nk; k++)
  //           { // C[i][j] += A[i][k]*B[j][k];
  //             sum1 += A[i * Nk + k] * B[j * Nk + k];
  //             // sum2 += A[(i + 1) * Nk + k] * B[j * Nk + k];
  //             // sum5 += A[i * Nk + k] * B[(j + 1) * Nk + k];
  //             // sum6 += A[(i + 1) * Nk + k] * B[(j + 1) * Nk + k];
  //             // sum3 += A[(i + 2) * Nk + k] * B[j * Nk + k];
  //             // sum4 += A[(i + 3) * Nk + k] * B[j * Nk + k];
  //           }
  //           C[i * Nj + j] = sum1;
  //           // C[(i + 1) * Nj + j] = sum2;
  //           // C[(i + 2) * Nj + j] = sum3;
  //           // C[(i + 3) * Nj + j] = sum4;
  //           // C[i * Nj + j + 1] = sum5;
  //           // C[(i + 1) * Nj + j + 1] = sum6;
  //         }
  // #pragma omp parallel for schedule(dynamic) private(i, j, k) reduction(+ : sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15, sum16)
  //   for (i = 0; i < Ni; i += 4)
  //     for (j = 0; j < Nj; j += 4)
  //     {
  //       sum1 = 0;
  //       sum2 = 0;
  //       sum3 = 0;
  //       sum4 = 0;
  //       sum5 = 0;
  //       sum6 = 0;
  //       sum7 = 0;
  //       sum8 = 0;
  //       sum9 = 0;
  //       sum10 = 0;
  //       sum11 = 0;
  //       sum12 = 0;
  //       sum13 = 0;
  //       sum14 = 0;
  //       sum15 = 0;
  //       sum16 = 0;
  //       for (k = 0; k < Nk; k++)
  //       {
  //         // C[i][j] += A[i][k]*B[j][k];
  //         sum1 += A[i * Nk + k] * B[j * Nk + k];
  //         sum2 += A[(i + 1) * Nk + k] * B[j * Nk + k];
  //         sum3 += A[(i + 2) * Nk + k] * B[j * Nk + k];
  //         sum4 += A[(i + 3) * Nk + k] * B[j * Nk + k];

  //         sum5 += A[i * Nk + k] * B[(j + 1) * Nk + k];
  //         sum6 += A[(i + 1) * Nk + k] * B[(j + 1) * Nk + k];
  //         sum7 += A[(i + 2) * Nk + k] * B[(j + 1) * Nk + k];
  //         sum8 += A[(i + 3) * Nk + k] * B[(j + 1) * Nk + k];

  //         sum9 += A[i * Nk + k] * B[(j + 2) * Nk + k];
  //         sum10 += A[(i + 1) * Nk + k] * B[(j + 2) * Nk + k];
  //         sum11 += A[(i + 2) * Nk + k] * B[(j + 2) * Nk + k];
  //         sum12 += A[(i + 3) * Nk + k] * B[(j + 2) * Nk + k];

  //         sum13 += A[i * Nk + k] * B[(j + 3) * Nk + k];
  //         sum14 += A[(i + 1) * Nk + k] * B[(j + 3) * Nk + k];
  //         sum15 += A[(i + 2) * Nk + k] * B[(j + 3) * Nk + k];
  //         sum16 += A[(i + 3) * Nk + k] * B[(j + 3) * Nk + k];
  //       }
  //       C[i * Nj + j] = sum1;
  //       C[(i + 1) * Nj + j] = sum2;
  //       C[(i + 2) * Nj + j] = sum3;
  //       C[(i + 3) * Nj + j] = sum4;

  //       C[i * Nj + j + 1] = sum5;
  //       C[(i + 1) * Nj + j + 1] = sum6;
  //       C[(i + 2) * Nj + j + 1] = sum7;
  //       C[(i + 3) * Nj + j + 1] = sum8;

  //       C[i * Nj + j + 2] = sum9;
  //       C[(i + 1) * Nj + j + 2] = sum10;
  //       C[(i + 2) * Nj + j + 2] = sum11;
  //       C[(i + 3) * Nj + j + 2] = sum12;

  //       C[i * Nj + j + 3] = sum13;
  //       C[(i + 1) * Nj + j + 3] = sum14;
  //       C[(i + 2) * Nj + j + 3] = sum15;
  //       C[(i + 3) * Nj + j + 3] = sum16;
  //     }

  // #pragma omp parallel for schedule(dynamic) reduction(+ : sum1)
  //   for (i = 0; i < Ni; i++)
  //   {
  //     for (j = 0; j < Nj; j++)
  //     {
  //       sum1 = 0;
  //       for (k = 0; k < Nk; k += 4)
  //       {
  //         // C[i][j] = C[i][j] + A[i][k]*B[j][k];
  //         sum1 += A[i * Nk + k] * B[j * Nk + k];
  //         sum1 += A[i * Nk + k + 1] * B[j * Nk + k + 1];
  //         sum1 += A[i * Nk + k + 2] * B[j * Nk + k + 2];
  //         sum1 += A[i * Nk + k + 3] * B[j * Nk + k + 3];
  //       }
  //       C[i * Nj + j] = sum1;
  //     }
  //   }

  // #pragma omp parallel for schedule(dynamic)
  //   for (i = 0; i < Ni; i += 4)
  //   {
  //     for (j = 0; j < Nj; j++)
  //     {
  //       sum1 = 0;
  //       sum2 = 0;
  //       sum3 = 0;
  //       sum4 = 0;
  //       for (k = 0; k < Nk; k++)
  //       {
  //         // C[i][j] = C[i][j] + A[i][k]*B[j][k];
  //         sum1 += A[i * Nk + k] * B[j * Nk + k];
  //         sum2 += A[(i + 1) * Nk + k] * B[j * Nk + k];
  //         sum3 += A[(i + 2) * Nk + k] * B[j * Nk + k];
  //         sum4 += A[(i + 3) * Nk + k] * B[j * Nk + k];
  //       }
  //       C[i * Nj + j] = sum1;
  //       C[(i + 1) * Nj + j] = sum2;
  //       C[(i + 2) * Nj + j] = sum3;
  //       C[(i + 3) * Nj + j] = sum4;
  //     }
  //   }

  //   for (i = 0; i < Ni; i++)
  //     for (j = 0; j < Nj; j++)
  //     {
  //       sum1 = 0;
  //       // #pragma omp simd
  //       // #pragma omp parallel for schedule(dynamic) firstprivate(i, j)
  // #pragma omp parallel for schedule(dynamic) shared(i, j)
  //       for (k = 0; k < Nk; k++)
  //       { // C[i][j] = C[i][j] + A[i][k]*B[j][k];
  //         sum1 += A[i * Nk + k] * B[j * Nk + k];
  //       }
  //       C[i * Nj + j] = sum1;
  //     }

  // #pragma omp parallel for schedule(dynamic) shared(i, j, C, sum1)
  //   for (k = 0; k < Nk; k += 4)
  //   {
  //     for (i = 0; i < Ni; i++)
  //     {
  //       sum1 = 0;
  //       for (j = 0; j < Nj; j++)
  //       {
  //         // C[i][j] = C[i][j] + A[i][k]*B[j][k];
  //         sum1 += A[i * Nk + k] * B[j * Nk + k];
  //         sum1 += A[i * Nk + k + 1] * B[j * Nk + k + 1];
  //         sum1 += A[i * Nk + k + 2] * B[j * Nk + k + 2];
  //         sum1 += A[i * Nk + k + 3] * B[j * Nk + k + 3];
  //       }
  //       C[i * Nj + j] = sum1;
  //     }
  //   }

  //   for (i = 0; i < Ni; i += 4)
  //   {
  //     for (j = 0; j < Nj; j++)
  //     {
  //       sum1 = 0;
  //       sum2 = 0;
  //       sum3 = 0;
  //       sum4 = 0;
  // #pragma omp parallel for schedule(dynamic)
  //       for (k = 0; k < Nk; k ++)
  //       {
  //         // C[i][j] = C[i][j] + A[i][k]*B[j][k];
  //         sum1 += A[i * Nk + k] * B[j * Nk + k];
  //         sum2 += A[(i + 1) * Nk + k] * B[j * Nk + k];
  //         sum3 += A[(i + 2) * Nk + k] * B[j * Nk + k];
  //         sum4 += A[(i + 3) * Nk + k] * B[j * Nk + k];
  //       }
  //       C[i * Nj + j] = sum1;
  //       C[(i + 1) * Nj + j] = sum2;
  //       C[(i + 2) * Nj + j] = sum3;
  //       C[(i + 3) * Nj + j] = sum4;
  //     }
  //   }

  //   for (i = 0; i < Ni; i++)
  //   {
  //     for (j = 0; j < Nj; j++)
  // #pragma omp parallel for schedule(dynamic)
  //       for (k = 0; k < Nk; k++)
  //       {
  //         {
  //           // C[i][j] = C[i][j] + A[i][k]*B[j][k];
  //           sum1 += A[i * Nk + k] * B[j * Nk + k];
  //           sum2 += A[(i + 1) * Nk + k] * B[j * Nk + k];
  //           sum3 += A[(i + 2) * Nk + k] * B[j * Nk + k];
  //           sum4 += A[(i + 3) * Nk + k] * B[j * Nk + k];
  //         }
  //         C[i * Nj + j] = sum1;
  //       }
  //   }

  // #pragma omp parallel for schedule(dynamic) private(i, j, k)
  //   for (i = 0; i < Ni; i++)
  //     for (j = 0; j < Nj; j++)
  //     {
  //       for (k = 0; k < Nk; k++)
  //       {
  //         // C[i][j] = C[i][j] + A[i][k]*B[j][k];
  //         C[i * Nj + j] += A[i * Nk + k] * B[j * Nk + k];
  //       }
  //       // C[i * Nj + j] = sum1;
  //     }
}

void aTbT_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;
#pragma omp parallel for schedule(dynamic) private(i, j, k)
  for (i = 0; i < Ni; i++)
    for (j = 0; j < Nj; j++)
      for (k = 0; k < Nk; k++)
        // C[i][j] += A[k][i]*B[j][k];
        C[i * Nj + j] += A[k * Ni + i] * B[j * Nk + k];
}
