#include <stdio.h>
#include <time.h>
#include <math.h>
#define threshold 0.0001
#define BLOCK_SIZE 16
#define FIXME 1

void checkCUDAError(const char *msg);

cudaEvent_t start, stop;
float tstart, elapsedTime;

__global__ void ab_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
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
     C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[k*Nj+j];
}

void abT_seq(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  for (i = 0; i < Ni; i++)
   for (j = 0; j < Nj; j++)
    for (k = 0; k < Nk; k++)
// C[i][j] = C[i][j] + A[i][k]*B[j][k];
     C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
}

void aTb_seq(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  for (i = 0; i < Ni; i++)
   for (j = 0; j < Nj; j++)
    for (k = 0; k < Nk; k++)
// C[i][j] = C[i][j] + A[k][i]*B[k][j];
     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
}

void aTbT_seq(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  for (i = 0; i < Ni; i++)
   for (j = 0; j < Nj; j++)
    for (k = 0; k < Nk; k++)
// C[i][j] = C[i][j] + A[k][i]*B[j][k];
     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
}


int main(){

  float *h_A, *h_B, *h_C, *h_Cref, *d_A, *d_B, *d_C;
  int i,j,k;
  int Ni,Nj,Nk;


  printf("Specify Matrix dimension Ni, Nj, Nk: ");
  scanf("%d %d %d", &Ni,&Nj,&Nk);

  h_A = (float *) malloc(sizeof(float)*Ni*Nk);
  h_B = (float *) malloc(sizeof(float)*Nk*Nj);
  h_C = (float *) malloc(sizeof(float)*Ni*Nj);
  h_Cref = (float *) malloc(sizeof(float)*Ni*Nj);;

  for (i=0; i<Ni; i++)
   for (k=0; k<Nk; k++)
    h_A[k*Ni+i] = rand();
  for (k=0; k<Nk; k++)
   for (j=0; j<Nj; j++)
    h_B[k*Nj+j] = rand();

  
 // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, Ni*Nk*sizeof(float));
  cudaMalloc(&d_B, Nk*Nj*sizeof(float));
  cudaMalloc(&d_C, Ni*Nj*sizeof(float));
  checkCUDAError("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, Ni*Nk*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, Nk*Nj*sizeof(float), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy H2D transfer failure");

  dim3 block(FIXME,FIXME);  
  dim3 grid(FIXME,FIXME);
  for(int version=0; version<4; version++)
  {
   for(i=0;i<Ni;i++) for(j=0;j<Nj;j++) h_Cref[i*Nj+j] = 0;
   switch (version) {
      case 0: ab_seq(h_A,h_B,h_Cref,Ni,Nj,Nk);  break;
      case 1: aTb_seq(h_A,h_B,h_Cref,Ni,Nj,Nk); break;
      case 2: abT_seq(h_A,h_B,h_Cref,Ni,Nj,Nk); break;
      case 3: aTbT_seq(h_A,h_B,h_Cref,Ni,Nj,Nk);
    }
    for(int trial=0;trial<3;trial++)
    {
     for(i=0;i<Ni;i++) for(j=0;j<Nj;j++) h_C[i*Nj+j] = 0; 
      printf("Trial %d: ",trial);
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);
      // Launch kernel
      switch (version) {
      case 0: ab_gpu<<<grid, block>>>(d_A, d_B, d_C,Ni,Nj,Nk); printf("AB "); break;
      case 1: aTb_gpu<<<grid, block>>>(d_A, d_B, d_C,Ni,Nj,Nk); printf("ATB ");break;
      case 2: abT_gpu<<<grid, block>>>(d_A, d_B, d_C,Ni,Nj,Nk); printf("ABT ");break;
      case 3: aTbT_gpu<<<grid, block>>>(d_A, d_B, d_C,Ni,Nj,Nk); printf("ATBT ");
      }
      checkCUDAError("GPU kernel launch failure");
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsedTime, start,stop);
      cudaDeviceSynchronize();
      // Copy results back to host
      cudaMemcpy(h_C, d_C, Ni*Nj*sizeof(float), cudaMemcpyDeviceToHost);
      checkCUDAError("cudaMemcpy D2H");
      for (int i = 0; i < Ni*Nj; i++) if (fabs((h_C[i]-h_Cref[i])/h_Cref[i])>threshold) {printf("Error: mismatch at linearized index %d, was: %f, should be: %f\n", i, h_C[i], h_Cref[i]); return -1;}
      printf("GFLOPS: %.2f\n",2.0e-6*Ni*Nj*Nk/elapsedTime);
     }
  }
  return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

