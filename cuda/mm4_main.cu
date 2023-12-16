#include <stdio.h>
#include <time.h>
#include <math.h>
#define threshold 0.0001
#define BLOCK_SIZE 16

void checkCUDAError(const char *msg);

cudaEvent_t start, stop;
float tstart, elapsedTime;

__global__ void ab_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
__global__ void ab16_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
__global__ void ab_gpu_1(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);

__global__ void aTb_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
__global__ void aTb16_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
__global__ void aTb_gpu_1(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);

__global__ void abT_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
__global__ void abT16_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
__global__ void abT_gpu_1(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);

__global__ void aTbT_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
__global__ void aTbT16_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
__global__ void aTbT_gpu_1(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);

__global__ void abT_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
__global__ void aTb_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
__global__ void aTbT_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);

void ab_seq(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  for (i = 0; i < Ni; i++)
    for (j = 0; j < Nj; j++)
      for (k = 0; k < Nk; k++)
        // C[i][j] = C[i][j] + A[i][k]*B[k][j];
        C[i * Nj + j] = C[i * Nj + j] + A[i * Nk + k] * B[k * Nj + j];
}

void abT_seq(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  for (i = 0; i < Ni; i++)
    for (j = 0; j < Nj; j++)
      for (k = 0; k < Nk; k++)
        // C[i][j] = C[i][j] + A[i][k]*B[j][k];
        C[i * Nj + j] = C[i * Nj + j] + A[i * Nk + k] * B[j * Nk + k];
}

void aTb_seq(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  for (i = 0; i < Ni; i++)
    for (j = 0; j < Nj; j++)
      for (k = 0; k < Nk; k++)
        // C[i][j] = C[i][j] + A[k][i]*B[k][j];
        C[i * Nj + j] = C[i * Nj + j] + A[k * Ni + i] * B[k * Nj + j];
}

void aTbT_seq(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  for (i = 0; i < Ni; i++)
    for (j = 0; j < Nj; j++)
      for (k = 0; k < Nk; k++)
        // C[i][j] = C[i][j] + A[k][i]*B[j][k];
        C[i * Nj + j] = C[i * Nj + j] + A[k * Ni + i] * B[j * Nk + k];
}

void ab_launch(float *d_A, float *d_B, float *d_C, int Ni, int Nj, int Nk)
{
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  if (Nk % 2 == 0)
    if ((Ni > 64) && (Nj > 64))
    {
      dim3 grid(ceil(Ni / (4 * float(BLOCK_SIZE))), ceil(Nj / (4 * float(BLOCK_SIZE))));
      // printf("Case 1: Block size (%d, %d); Grid size (%d, %d)\n", block.x, block.y, grid.x, grid.y);
      ab_gpu<<<grid, block>>>(d_A, d_B, d_C, Ni, Nj, Nk);
    }
    else
    {
      dim3 grid(ceil(Ni / float(BLOCK_SIZE)), ceil(Nj / float(BLOCK_SIZE)));
      // printf("Case 2: Block size (%d, %d); Grid size (%d, %d)\n", block.x, block.y, grid.x, grid.y);
      ab16_gpu<<<grid, block>>>(d_A, d_B, d_C, Ni, Nj, Nk);
    }
  else
  {
    if ((Ni > 64) && (Nj > 64))
    {
      dim3 grid(ceil(Ni / float(BLOCK_SIZE)), ceil(Nj / float(BLOCK_SIZE)));
      // printf("Case 3: Block size (%d, %d); Grid size (%d, %d)\n", block.x, block.y, grid.x, grid.y);
      ab_gpu_1<<<grid, block>>>(d_A, d_B, d_C, Ni, Nj, Nk);
    }
    else
    {
      dim3 grid(ceil(Ni / float(BLOCK_SIZE)), ceil(Nj / float(BLOCK_SIZE)));
      // printf("Case 4: Block size (%d, %d); Grid size (%d, %d)\n", block.x, block.y, grid.x, grid.y);
      ab_gpu_1<<<grid, block>>>(d_A, d_B, d_C, Ni, Nj, Nk);
    }
  }
}

void aTb_launch(float *d_A, float *d_B, float *d_C, int Ni, int Nj, int Nk)
{
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  if (Nk % 2 == 0)
    if ((Ni > 64) && (Nj > 64))
    {
      dim3 grid(ceil(Ni / (4 * float(BLOCK_SIZE))), ceil(Nj / (4 * float(BLOCK_SIZE))));
      // printf("Case 1: Block size (%d, %d); Grid size (%d, %d)\n", block.x, block.y, grid.x, grid.y);
      aTb_gpu<<<grid, block>>>(d_A, d_B, d_C, Ni, Nj, Nk);
    }
    else
    {
      dim3 grid(ceil(Ni / float(BLOCK_SIZE)), ceil(Nj / float(BLOCK_SIZE)));
      // printf("Case 2: Block size (%d, %d); Grid size (%d, %d)\n", block.x, block.y, grid.x, grid.y);
      aTb16_gpu<<<grid, block>>>(d_A, d_B, d_C, Ni, Nj, Nk);
    }
  else
  {
    if ((Ni > 64) && (Nj > 64))
    {
      dim3 grid(ceil(Ni / float(BLOCK_SIZE)), ceil(Nj / float(BLOCK_SIZE)));
      // printf("Case 3: Block size (%d, %d); Grid size (%d, %d)\n", block.x, block.y, grid.x, grid.y);
      aTb_gpu_1<<<grid, block>>>(d_A, d_B, d_C, Ni, Nj, Nk);
    }
    else
    {
      dim3 grid(ceil(Ni / float(BLOCK_SIZE)), ceil(Nj / float(BLOCK_SIZE)));
      // printf("Case 4: Block size (%d, %d); Grid size (%d, %d)\n", block.x, block.y, grid.x, grid.y);
      aTb_gpu_1<<<grid, block>>>(d_A, d_B, d_C, Ni, Nj, Nk);
    }
  }
}

void abT_launch(float *d_A, float *d_B, float *d_C, int Ni, int Nj, int Nk)
{
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  if (Nk % 2 == 0)
    if ((Ni > 64) && (Nj > 64))
    {
      dim3 grid(ceil(Ni / (4 * float(BLOCK_SIZE))), ceil(Nj / (4 * float(BLOCK_SIZE))));
      // printf("Case 1: Block size (%d, %d); Grid size (%d, %d)\n", block.x, block.y, grid.x, grid.y);
      abT_gpu<<<grid, block>>>(d_A, d_B, d_C, Ni, Nj, Nk);
    }
    else
    {
      dim3 grid(ceil(Ni / float(BLOCK_SIZE)), ceil(Nj / float(BLOCK_SIZE)));
      // printf("Case 2: Block size (%d, %d); Grid size (%d, %d)\n", block.x, block.y, grid.x, grid.y);
      abT16_gpu<<<grid, block>>>(d_A, d_B, d_C, Ni, Nj, Nk);
    }
  else
  {
    if ((Ni > 64) && (Nj > 64))
    {
      dim3 grid(ceil(Ni / float(BLOCK_SIZE)), ceil(Nj / float(BLOCK_SIZE)));
      printf("Case 3: Block size (%d, %d); Grid size (%d, %d)\n", block.x, block.y, grid.x, grid.y);
      abT_gpu_1<<<grid, block>>>(d_A, d_B, d_C, Ni, Nj, Nk);
    }
    else
    {
      dim3 grid(ceil(Ni / float(BLOCK_SIZE)), ceil(Nj / float(BLOCK_SIZE)));
      // printf("Case 4: Block size (%d, %d); Grid size (%d, %d)\n", block.x, block.y, grid.x, grid.y);
      abT_gpu_1<<<grid, block>>>(d_A, d_B, d_C, Ni, Nj, Nk);
    }
  }
}

void aTbT_launch(float *d_A, float *d_B, float *d_C, int Ni, int Nj, int Nk)
{
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  if (Nk % 2 == 0)
    if ((Ni > 64) && (Nj > 64))
    {
      dim3 grid(ceil(Ni / (4 * float(BLOCK_SIZE))), ceil(Nj / (4 * float(BLOCK_SIZE))));
      // printf("aTbT Case 1: Block size (%d, %d); Grid size (%d, %d)\n", block.x, block.y, grid.x, grid.y);
      aTbT_gpu<<<grid, block>>>(d_A, d_B, d_C, Ni, Nj, Nk);
    }
    else
    {
      dim3 grid(ceil(Ni / float(BLOCK_SIZE)), ceil(Nj / float(BLOCK_SIZE)));
      // printf("aTbT Case 2: Block size (%d, %d); Grid size (%d, %d)\n", block.x, block.y, grid.x, grid.y);
      aTbT16_gpu<<<grid, block>>>(d_A, d_B, d_C, Ni, Nj, Nk);
    }
  else
  {
    if ((Ni > 64) && (Nj > 64))
    {
      dim3 grid(ceil(Ni / float(BLOCK_SIZE)), ceil(Nj / float(BLOCK_SIZE)));
      // printf("Case 3: Block size (%d, %d); Grid size (%d, %d)\n", block.x, block.y, grid.x, grid.y);
      aTbT_gpu_1<<<grid, block>>>(d_A, d_B, d_C, Ni, Nj, Nk);
    }
    else
    {
      dim3 grid(ceil(Ni / float(BLOCK_SIZE)), ceil(Nj / float(BLOCK_SIZE)));
      // printf("Case 4: Block size (%d, %d); Grid size (%d, %d)\n", block.x, block.y, grid.x, grid.y);
      aTbT_gpu_1<<<grid, block>>>(d_A, d_B, d_C, Ni, Nj, Nk);
    }
  }
}

int main(int argc, char *argv[])
{
  float *h_A, *h_B, *h_C, *h_Cref, *d_A, *d_B, *d_C;
  int i, j, k;
  int Ni, Nj, Nk;

  // printf("Specify Matrix dimension Ni, Nj, Nk: ");
  // scanf("%d %d %d", &Ni,&Nj,&Nk);
  if (argc != 4)
  {
    printf("Invalid # of args. Expected 3 arguments Ni, Nj, Nk.\n");
    exit(1);
  }

  Ni = atoi(argv[1]);
  Nj = atoi(argv[2]);
  Nk = atoi(argv[3]);
  printf("_______________________________________________________________________________________\n");
  printf("Matrix dimension Ni: %d, Nj %d, Nk: %d\n", Ni, Nj, Nk);

  h_A = (float *)malloc(sizeof(float) * Ni * Nk);
  h_B = (float *)malloc(sizeof(float) * Nk * Nj);
  h_C = (float *)malloc(sizeof(float) * Ni * Nj);
  h_Cref = (float *)malloc(sizeof(float) * Ni * Nj);

  for (i = 0; i < Ni; i++)
    for (k = 0; k < Nk; k++)
      h_A[k * Ni + i] = rand();

  for (k = 0; k < Nk; k++)
    for (j = 0; j < Nj; j++)
      h_B[k * Nj + j] = rand();

  // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, Ni * Nk * sizeof(float));
  cudaMalloc(&d_B, Nk * Nj * sizeof(float));
  cudaMalloc(&d_C, Ni * Nj * sizeof(float));
  checkCUDAError("cudaMalloc failure");

  cudaMemcpy(d_A, h_A, Ni * Nk * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, Nk * Nj * sizeof(float), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy H2D transfer failure");

  for (int version = 0; version < 4; version++)
  {
    for (i = 0; i < Ni; i++)
      for (j = 0; j < Nj; j++)
        h_Cref[i * Nj + j] = 0;

    // version = 3;
    switch (version)
    {
    case 0:
      ab_seq(h_A, h_B, h_Cref, Ni, Nj, Nk);
      break;
    case 1:
      aTb_seq(h_A, h_B, h_Cref, Ni, Nj, Nk);
      break;
    case 2:
      abT_seq(h_A, h_B, h_Cref, Ni, Nj, Nk);
      break;
    case 3:
      aTbT_seq(h_A, h_B, h_Cref, Ni, Nj, Nk);
      break;
    }

    for (int trial = 0; trial < 3; trial++)
    {
      if (trial == 0)
        printf("\n");

      for (i = 0; i < Ni; i++)
        for (j = 0; j < Nj; j++)
          h_C[i * Nj + j] = 0;

      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);

      // Launch kernel
      switch (version)
      {
      case 0:
        ab_launch(d_A, d_B, d_C, Ni, Nj, Nk);
        printf("Trial %d: AB ", trial);
        break;
      case 1:
        aTb_launch(d_A, d_B, d_C, Ni, Nj, Nk);
        printf("Trial %d: ATB ", trial);
        break;
      case 2:
        abT_launch(d_A, d_B, d_C, Ni, Nj, Nk);
        printf("Trial %d: ABT ", trial);
        break;
      case 3:
        aTbT_launch(d_A, d_B, d_C, Ni, Nj, Nk);
        printf("Trial %d: ATBT ", trial);
        break;
      }
      checkCUDAError("GPU kernel launch failure");
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsedTime, start, stop);
      cudaDeviceSynchronize();

      // Copy results back to host
      cudaMemcpy(h_C, d_C, Ni * Nj * sizeof(float), cudaMemcpyDeviceToHost);
      checkCUDAError("cudaMemcpy D2H");
      for (int i = 0; i < Ni * Nj; i++)
        if (fabs((h_C[i] - h_Cref[i]) / h_Cref[i]) > threshold)
        {
          printf("Error: mismatch at linearized index %d, was: %f, should be: %f\n", i, h_C[i], h_Cref[i]);
          printf("\nOutput\n\n");
          int DSIZE = 32;
          for (int p = 0; p < DSIZE * DSIZE; p++)
          {
            printf("%.0f  ", h_C[p]);
            if (p != 0 && p % DSIZE == 0)
              printf("\n");
          }
          printf("\n");
          printf("\nReference\n");
          printf("\n");
          for (int p = 0; p < DSIZE * DSIZE; p++)
          {
            printf("%.0f  ", h_Cref[p]);
            if (p != 0 && p % DSIZE == 0)
              printf("\n");
          }
          return -1;
        }
      printf("GFLOPS: %.2f\n", 2.0e-6 * Ni * Nj * Nk / elapsedTime);
    }
    // return 0;
  }
  return 0;
}

void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    fprintf(stderr, "Cuda error: %s, Reason: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
