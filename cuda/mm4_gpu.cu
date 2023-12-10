__global__ void ab_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
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
