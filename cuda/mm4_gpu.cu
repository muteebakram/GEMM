#define BLOCK_SIZE 16

__global__ void ab_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  // Coalesced memory access for A, B, C
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int i = 4 * (blockIdx.y * blockDim.y + ty);
  int j = 4 * blockIdx.x * blockDim.x + tx;

  if ((i < Ni) && (j < Nj))
  {
    // int BLOCK_SIZE = 32;
    float sum00 = 0, sum10 = 0, sum20 = 0, sum30 = 0;
    float sum01 = 0, sum11 = 0, sum21 = 0, sum31 = 0;
    float sum02 = 0, sum12 = 0, sum22 = 0, sum32 = 0;
    float sum03 = 0, sum13 = 0, sum23 = 0, sum33 = 0;
    __shared__ float sA1[BLOCK_SIZE][BLOCK_SIZE], sA2[BLOCK_SIZE][BLOCK_SIZE], sA3[BLOCK_SIZE][BLOCK_SIZE], sA4[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB1[BLOCK_SIZE][BLOCK_SIZE], sB2[BLOCK_SIZE][BLOCK_SIZE], sB3[BLOCK_SIZE][BLOCK_SIZE], sB4[BLOCK_SIZE][BLOCK_SIZE];

    for (int ks = 0; ks < Nk; ks += BLOCK_SIZE)
    {
      sA1[ty][tx] = A[i * Nk + ks + tx];
      sA2[ty][tx] = A[(i + 1) * Nk + ks + tx];
      sA3[ty][tx] = A[(i + 2) * Nk + ks + tx];
      sA4[ty][tx] = A[(i + 3) * Nk + ks + tx];

      sB1[ty][tx] = B[(ks + ty) * Nj + j];
      sB2[ty][tx] = B[(ks + ty) * Nj + j + BLOCK_SIZE];
      sB3[ty][tx] = B[(ks + ty) * Nj + j + 2 * BLOCK_SIZE];
      sB4[ty][tx] = B[(ks + ty) * Nj + j + 3 * BLOCK_SIZE];
      __syncthreads();

      for (int k = 0; k < BLOCK_SIZE; k++)
      {
        // C[i][j] = C[i][j] + A[i][k]*B[k][j];
        sum00 += sA1[ty][k] * sB1[k][tx];
        sum10 += sA2[ty][k] * sB1[k][tx];
        sum20 += sA3[ty][k] * sB1[k][tx];
        sum30 += sA4[ty][k] * sB1[k][tx];

        sum01 += sA1[ty][k] * sB2[k][tx];
        sum11 += sA2[ty][k] * sB2[k][tx];
        sum21 += sA3[ty][k] * sB2[k][tx];
        sum31 += sA4[ty][k] * sB2[k][tx];

        sum02 += sA1[ty][k] * sB3[k][tx];
        sum12 += sA2[ty][k] * sB3[k][tx];
        sum22 += sA3[ty][k] * sB3[k][tx];
        sum32 += sA4[ty][k] * sB3[k][tx];

        sum03 += sA1[ty][k] * sB4[k][tx];
        sum13 += sA2[ty][k] * sB4[k][tx];
        sum23 += sA3[ty][k] * sB4[k][tx];
        sum33 += sA4[ty][k] * sB4[k][tx];
      }

      __syncthreads();
    }
    C[i * Nj + j] = sum00;
    C[(i + 1) * Nj + j] = sum10;
    C[(i + 2) * Nj + j] = sum20;
    C[(i + 3) * Nj + j] = sum30;

    C[i * Nj + j + BLOCK_SIZE] = sum01;
    C[(i + 1) * Nj + j + BLOCK_SIZE] = sum11;
    C[(i + 2) * Nj + j + BLOCK_SIZE] = sum21;
    C[(i + 3) * Nj + j + BLOCK_SIZE] = sum31;

    C[i * Nj + j + 2 * BLOCK_SIZE] = sum02;
    C[(i + 1) * Nj + j + 2 * BLOCK_SIZE] = sum12;
    C[(i + 2) * Nj + j + 2 * BLOCK_SIZE] = sum22;
    C[(i + 3) * Nj + j + 2 * BLOCK_SIZE] = sum32;

    C[i * Nj + j + 3 * BLOCK_SIZE] = sum03;
    C[(i + 1) * Nj + j + 3 * BLOCK_SIZE] = sum13;
    C[(i + 2) * Nj + j + 3 * BLOCK_SIZE] = sum23;
    C[(i + 3) * Nj + j + 3 * BLOCK_SIZE] = sum33;
  }
}

__global__ void ab16_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  // Coalesced memory access for A, B, C
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int i = blockIdx.y * blockDim.y + ty;
  int j = blockIdx.x * blockDim.x + tx;

  if ((i < Ni) && (j < Nj))
  {
    float sum00 = 0;
    __shared__ float sA1[BLOCK_SIZE][16 * BLOCK_SIZE], sB1[16 * BLOCK_SIZE][BLOCK_SIZE];

    for (int ks = 0; ks < Nk; ks += 16 * BLOCK_SIZE)
    {
      int indexA = i * Nk + ks + tx;
      for (int m = 0; m < 16; m++)
      {
        sA1[ty][tx + m * BLOCK_SIZE] = A[indexA + m * BLOCK_SIZE];
        sB1[ty + m * BLOCK_SIZE][tx] = B[(ks + m * BLOCK_SIZE + ty) * Nj + j];
      }
      __syncthreads();

      for (int k = 0; k < 16 * BLOCK_SIZE; k += 8)
      {
        // C[i][j] = C[i][j] + A[i][k]*B[k][j];
        sum00 += sA1[ty][k] * sB1[k][tx];
        sum00 += sA1[ty][k + 1] * sB1[k + 1][tx];
        sum00 += sA1[ty][k + 2] * sB1[k + 2][tx];
        sum00 += sA1[ty][k + 3] * sB1[k + 3][tx];

        sum00 += sA1[ty][k + 4] * sB1[k + 4][tx];
        sum00 += sA1[ty][k + 5] * sB1[k + 5][tx];
        sum00 += sA1[ty][k + 6] * sB1[k + 6][tx];
        sum00 += sA1[ty][k + 7] * sB1[k + 7][tx];
      }
      __syncthreads();
    }

    C[i * Nj + j] = sum00;
  }
}

__global__ void ab_gpu_1(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if ((i < Ni) && (j < Nj))
  {
    float sum00 = 0;
    for (int k = 0; k < Nk; k++)
      // C[i][j] = C[i][j] + A[i][k]*B[k][j];
      sum00 += A[i * Nk + k] * B[k * Nj + j];
    C[i * Nj + j] = sum00;
  }
}

__global__ void abT_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  for (i = 0; i < Ni; i++)
    for (j = 0; j < Nj; j++)
      C[i * Nj + j] = 0.0;

  for (i = 0; i < Ni; i++)
    for (j = 0; j < Nj; j++)
      for (k = 0; k < Nk; k++)
        // C[i][j] = C[i][j] + A[i][k]*B[j][k];
        C[i * Nj + j] = C[i * Nj + j] + A[i * Nk + k] * B[j * Nk + k];
}

__global__ void aTb_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  for (i = 0; i < Ni; i++)
    for (j = 0; j < Nj; j++)
      C[i * Nj + j] = 0.0;

  for (i = 0; i < Ni; i++)
    for (j = 0; j < Nj; j++)
      for (k = 0; k < Nk; k++)
        // C[i][j] = C[i][j] + A[k][i]*B[k][j];
        C[i * Nj + j] = C[i * Nj + j] + A[k * Ni + i] * B[k * Nj + j];
}

__global__ void aTbT_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  for (i = 0; i < Ni; i++)
    for (j = 0; j < Nj; j++)
      C[i * Nj + j] = 0.0;

  for (i = 0; i < Ni; i++)
    for (j = 0; j < Nj; j++)
      for (k = 0; k < Nk; k++)
        // C[i][j] = C[i][j] + A[k][i]*B[j][k];
        C[i * Nj + j] = C[i * Nj + j] + A[k * Ni + i] * B[j * Nk + k];
}
