#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define n 53
#define num_matrices 1

extern "C"
{
#include "config.h"
}

extern double *read_matrix_from_csv_flat(const char *filename, int rows, int cols);

// Error metric functions
__host__ void e1(double *A, double *L, double *U, int *P)
{
    double max_error = 0.0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double pa = A[P[i] * n + j];
            double lu = 0.0;
            for (int k = 0; k < n; k++)
            {
                lu += L[i * n + k] * U[k * n + j];
            }
            double diff = fabs(pa - lu);
            if (diff > max_error)
                max_error = diff;
        }
    }
    printf("E1 (Decomposition error ∞-norm): %.6e\n", max_error);
}

__host__ void e2(double *x_computed, double *x_true)
{
    double max_diff = 0.0;
    double max_x_true = 0.0;
    for (int i = 0; i < n; i++)
    {
        double diff = fabs(x_computed[i] - x_true[i]);
        if (diff > max_diff)
            max_diff = diff;
        if (fabs(x_true[i]) > max_x_true)
            max_x_true = fabs(x_true[i]);
    }
    double relative_error = max_diff / (max_x_true + 1e-12); // avoid div by zero
    printf("E2 (Relative solution error ∞-norm): %.6e\n", relative_error);
}

__host__ void e3(double *A, double *x, double *b)
{
    double max_residual = 0.0;
    for (int i = 0; i < n; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < n; j++)
        {
            sum += A[i * n + j] * x[j];
        }
        double residual = fabs(sum - b[i]);
        if (residual > max_residual)
            max_residual = residual;
    }
    printf("E3 (Residual norm ∞-norm): %.6e\n", max_residual);
}

__device__ void device_swap_rows(double *A, int *P, int row1, int row2)
{
    for (int j = 0; j < n; j++)
    {
        double temp = A[row1 * n + j];
        A[row1 * n + j] = A[row2 * n + j];
        A[row2 * n + j] = temp;
    }
    int temp_p = P[row1];
    P[row1] = P[row2];
    P[row2] = temp_p;
}

__global__ void lu_solve_kernel(double *A, int *P, double *b, double *x, double *L, double *U)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_matrices)
        return;

    double *local_A = &A[tid * n * n];
    int *local_P = &P[tid * n];
    double *local_b = &b[tid * n];
    double *local_x = &x[tid * n];
    double *local_L = &L[tid * n * n];
    double *local_U = &U[tid * n * n];

    // Initialize permutation vector
    for (int i = 0; i < n; i++)
        local_P[i] = i;

    // LU decomposition
    for (int k = 0; k < n - 1; k++)
    {
        // Partial pivoting
        int max_row = k;
        for (int i = k + 1; i < n; i++)
        {
            if (fabs(local_A[i * n + k]) > fabs(local_A[max_row * n + k]))
            {
                max_row = i;
            }
        }
        if (max_row != k)
        {
            device_swap_rows(local_A, local_P, k, max_row);
        }

        // Gaussian elimination
        for (int i = k + 1; i < n; i++)
        {
            local_A[i * n + k] /= local_A[k * n + k];
            for (int j = k + 1; j < n; j++)
            {
                local_A[i * n + j] -= local_A[i * n + k] * local_A[k * n + j];
            }
        }
    }

    // Extract L and U
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i > j)
            {
                local_L[i * n + j] = local_A[i * n + j];
                local_U[i * n + j] = 0.0;
            }
            else
            {
                local_U[i * n + j] = local_A[i * n + j];
                local_L[i * n + j] = (i == j) ? 1.0 : 0.0;
            }
        }
    }

    // Solve Ly = Pb
    double y[n];
    double pb[n];
    for (int i = 0; i < n; i++)
        pb[i] = local_b[local_P[i]];

    for (int i = 0; i < n; i++)
    {
        y[i] = pb[i];
        for (int j = 0; j < i; j++)
        {
            y[i] -= local_L[i * n + j] * y[j];
        }
    }

    // Solve Ux = y
    for (int i = n - 1; i >= 0; i--)
    {
        local_x[i] = y[i];
        for (int j = i + 1; j < n; j++)
        {
            local_x[i] -= local_U[i * n + j] * local_x[j];
        }
        local_x[i] /= local_U[i * n + i];
    }
}

int main()
{
    // Host memory allocations
    double *h_A, *h_b, *h_x_true, *h_x_computed, *h_L, *h_U;
    int *h_P;

    h_A = read_matrix_from_csv_flat("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/A_matrix_case1.csv", n, n);
    //  printf("File got read");
    //   h_A_copy = read_matrix_from_csv_flat("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/A_matrix_case1.csv", n, n);
    h_b = read_matrix_from_csv_flat("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/B_matrix_case1.csv", n, 1);
    h_x_true = read_matrix_from_csv_flat("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/Case_1_soln.csv", n, 1);

    h_x_computed = (double *)malloc(num_matrices * n * sizeof(double));
    h_L = (double *)malloc(num_matrices * n * n * sizeof(double));
    h_U = (double *)malloc(num_matrices * n * n * sizeof(double));
    h_P = (int *)malloc(num_matrices * n * sizeof(int));

    // Device memory allocations
    double *d_A, *d_b, *d_x, *d_L, *d_U;
    int *d_P;

    cudaMalloc(&d_A, num_matrices * n * n * sizeof(double));
    cudaMalloc(&d_b, num_matrices * n * sizeof(double));
    cudaMalloc(&d_x, num_matrices * n * sizeof(double));
    cudaMalloc(&d_L, num_matrices * n * n * sizeof(double));
    cudaMalloc(&d_U, num_matrices * n * n * sizeof(double));
    cudaMalloc(&d_P, num_matrices * n * sizeof(int));

    // Initialize matrices (Replace with actual data loading)
    // For demonstration, we'll use identity matrix and ones vector
    // for(int i = 0; i < n*n; i++) h_A[i] = (i/n == i%n) ? 1.0 : 0.0;
    // for(int i = 0; i < n; i++) h_b[i] = 1.0;
    // for(int i = 0; i < n; i++) h_x_true[i] = 1.0;

    // Copy to device
    cudaMemcpy(d_A, h_A, num_matrices * n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, num_matrices * n * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blocks(1);
    dim3 threads(1);
    lu_solve_kernel<<<blocks, threads>>>(d_A, d_P, d_b, d_x, d_L, d_U);

    // Copy results back
    cudaMemcpy(h_x_computed, d_x, num_matrices * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_L, d_L, num_matrices * n * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_U, d_U, num_matrices * n * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_P, d_P, num_matrices * n * sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate and print errors
    e1(h_A, h_L, h_U, h_P);     // E1: Decomposition error ∞-norm
    e2(h_x_computed, h_x_true); // E2: Relative solution error ∞-norm
    e3(h_A, h_x_computed, h_b); // E3: Residual norm ∞-norm

    // Cleanup
    free(h_A);
    free(h_b);
    free(h_x_true);
    free(h_x_computed);
    free(h_L);
    free(h_U);
    free(h_P);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_L);
    cudaFree(d_U);
    cudaFree(d_P);

    return 0;
}