#include <time.h>
#include <stdio.h>
#include <stdlib.h>

int Ni, Nj, Nk;
clock_t begin, end, beginRef, endRef;

void base(float *matA, float *matB, float *matCRef, float *matC)
{
    int i, j, k;

    beginRef = clock();
    for (i = 0; i < Ni; i++)
        for (j = 0; j < Nj; j++)
            for (k = 0; k < Nk; k++)
                matCRef[i * Nk + j] += matA[i * Nk + k] * matB[k * Nj + j];

    endRef = clock();

    begin = clock();
    for (i = 0; i < Ni; i++)
        for (j = 0; j < Nj; j++)
            for (k = 0; k < Nk; k++)
                matC[i * Nk + j] += matA[i * Nk + k] * matB[k * Nj + j];

    end = clock();
}

void t_nt(float *matA, float *matB, float *matCRef, float *matC)
{
    int i, j, k;

    beginRef = clock();
    for (i = 0; i < Ni; i++)
        for (j = 0; j < Nj; j++)
            for (k = 0; k < Nk; k++)
                matCRef[i * Nk + j] += matA[k * Ni + i] * matB[k * Nj + j];

    endRef = clock();

    begin = clock();
    for (i = 0; i < Ni; i++)
        for (j = 0; j < Nj; j++)
            for (k = 0; k < Nk; k++)
                matC[i * Nk + j] += matA[k * Ni + i] * matB[k * Nj + j];

    end = clock();
}

void nt_t(float *matA, float *matB, float *matCRef, float *matC)
{
    int i, j, k;

    beginRef = clock();
    for (i = 0; i < Ni; i++)
        for (j = 0; j < Nj; j++)
            for (k = 0; k < Nk; k++)
                matCRef[i * Nk + j] += matA[i * Nk + k] * matB[j * Nk + k];

    endRef = clock();

    begin = clock();
    for (i = 0; i < Ni; i++)
        for (j = 0; j < Nj; j++)
            for (k = 0; k < Nk; k++)
                matC[i * Nk + j] += matA[i * Nk + k] * matB[j * Nk + k];

    end = clock();
}

void t_t(float *matA, float *matB, float *matCRef, float *matC)
{
    int i, j, k;

    beginRef = clock();
    for (i = 0; i < Ni; i++)
        for (j = 0; j < Nj; j++)
            for (k = 0; k < Nk; k++)
                matCRef[i * Nk + j] += matA[i * Nk + k] * matB[j * Nk + k];

    endRef = clock();

    begin = clock();
    for (i = 0; i < Ni; i++)
        for (j = 0; j < Nj; j++)
            for (k = 0; k < Nk; k++)
                matC[i * Nk + j] += matA[i * Nk + k] * matB[j * Nk + k];

    end = clock();
}

void clean_up(float *matA, float *matB, float *matCRef, float *matC)
{
    free(matA);
    free(matB);
    free(matC);
    free(matCRef);
}

void validate(float *matA, float *matB, float *matCRef, float *matC)
{
    int i, j;
    for (i = 0; i < Ni; i++)
        for (j = 0; j < Nj; j++)
            if (matCRef[i * Nk + j] != matC[i * Nk + j])
            {
                printf("Mismatch at C[%d][%d]. MatCValue: %f, MatCRefValue: %f\n", i, j, matC[i * Nk + j], matCRef[i * Nk + j]);
                clean_up(matA, matB, matCRef, matC);
                exit(1);
            }
}

void performance_check()
{
    printf("Success\n");
    double time_before = (double)(endRef - beginRef) / CLOCKS_PER_SEC;
    double time_after = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time Before: %f secs, Time After : %f secs, Speedup: %.4f, GFLOPs: %.2f\n\n", time_before, time_after, time_before / time_after, (2e-9 * Ni * Nj) / time_after);
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        printf("Invalid # of args. Expected 4.\n");
        exit(1);
    }

    Ni = atoi(argv[1]);
    Nj = atoi(argv[2]);
    Nk = atoi(argv[3]);

    int i, j, k;
    float *matA = (float *)malloc(Ni * Nk * sizeof(float));
    float *matB = (float *)malloc(Nk * Nj * sizeof(float));
    float *matC = (float *)malloc(Ni * Nj * sizeof(float));
    float *matCRef = (float *)malloc(Ni * Nj * sizeof(float));

    for (i = 0; i < Ni; i++)
        for (k = 0; k < Nk; k++)
            matA[i * Nk + k] = rand() / (float)64;

    for (k = 0; k < Nk; k++)
        for (j = 0; j < Nj; j++)
            matB[k * Nj + j] = rand() / (float)64;

    printf("Performing base...\n");
    base(matA, matB, matCRef, matC);
    validate(matA, matB, matCRef, matC);
    performance_check();

    printf("Performing t_nt...\n");
    t_nt(matA, matB, matCRef, matC);
    validate(matA, matB, matCRef, matC);
    performance_check();

    printf("Performing nt_t...\n");
    nt_t(matA, matB, matCRef, matC);
    validate(matA, matB, matCRef, matC);
    performance_check();

    printf("Performing t_t...\n");
    t_t(matA, matB, matCRef, matC);
    validate(matA, matB, matCRef, matC);
    performance_check();

    clean_up(matA, matB, matCRef, matC);
    exit(0);
}