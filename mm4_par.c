
void ab_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
    int i, j, k;
    for (i = 0; i < Ni; i++)
        for (j = 0; j < Nj; j++)
            for (k = 0; k < Nk; k++)
                // C[i][j] = C[i][j] + A[i][k]*B[k][j];
                C[i * Nj + j] = C[i * Nj + j] + A[i * Nk + k] * B[k * Nj + j];
}

void abT_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
    int i, j, k;
    for (i = 0; i < Ni; i++)
        for (j = 0; j < Nj; j++)
            for (k = 0; k < Nk; k++)
                // C[i][j] = C[i][j] + A[i][k]*B[j][k];
                C[i * Nj + j] = C[i * Nj + j] + A[i * Nk + k] * B[j * Nk + k];
}

void aTb_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
    int i, j, k;
    for (i = 0; i < Ni; i++)
        for (j = 0; j < Nj; j++)
            for (k = 0; k < Nk; k++)
                // C[i][j] = C[i][j] + A[k][i]*B[k][j];
                C[i * Nj + j] = C[i * Nj + j] + A[k * Ni + i] * B[k * Nj + j];
}

void aTbT_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
    int i, j, k;
    for (i = 0; i < Ni; i++)
        for (j = 0; j < Nj; j++)
            for (k = 0; k < Nk; k++)
                // C[i][j] = C[i][j] + A[k][i]*B[j][k];
                C[i * Nj + j] = C[i * Nj + j] + A[k * Ni + i] * B[j * Nk + k];
}
