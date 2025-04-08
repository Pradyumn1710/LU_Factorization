#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

extern "C" {
    #include "config.h"
    }
    

__device__ void device_swap_rows(double* A, int* P, int row1, int row2, int n) {
    for (int j = 0; j < n; j++) {
        double temp = A[row1 * n + j];
        A[row1 * n + j] = A[row2 * n + j];
        A[row2 * n + j] = temp;
    }
    int temp_p = P[row1];
    P[row1] = P[row2];
    P[row2] = temp_p;
}

__global__ void lu_solve_kernel(double* all_A, int* all_P, double* all_b, double* all_x, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate offsets for this thread's data
    double* A = &all_A[tid * n * n];
    int* P = &all_P[tid * n];
    double* b = &all_b[tid * n];
    double* x = &all_x[tid * n];
    
    // Initialize permutation vector
    for (int i = 0; i < n; i++) P[i] = i;

    // LU Decomposition with partial pivoting
    for (int k = 0; k < n-1; k++) {
        // Partial pivoting
        int max_row = k;
        for (int i = k+1; i < n; i++) {
            if (fabs(A[i*n + k]) > fabs(A[max_row*n + k])) {
                max_row = i;
            }
        }
        if (max_row != k) {
            device_swap_rows(A, P, k, max_row, n);
        }

        // Gaussian elimination
        for (int i = k+1; i < n; i++) {
            A[i*n + k] /= A[k*n + k];
            for (int j = k+1; j < n; j++) {
                A[i*n + j] -= A[i*n + k] * A[k*n + j];
            }
        }
    }

    // Forward substitution (Ly = Pb)
    double y[53]; // Adjust size based on your 'n'
    
    // Apply permutation to b
    double pb[53];
    for (int i = 0; i < n; i++) pb[i] = b[P[i]];
    
    for (int i = 0; i < n; i++) {
        y[i] = pb[i];
        for (int j = 0; j < i; j++) {
            y[i] -= A[i*n + j] * y[j];
        }
    }

    // Backward substitution (Ux = y)
    for (int i = n-1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i+1; j < n; j++) {
            x[i] -= A[i*n + j] * x[j];
        }
        x[i] /= A[i*n + i];
    }
}

extern double* read_matrix_from_csv_flat(const char* filename, int rows, int cols);


int main() {
    const int n = 53; // Matrix size
    const int num_matrices = 1; // Number of matrices to process
    
    // Host allocations
    // double* h_A = (double*)malloc(num_matrices * n * n * sizeof(double));
    // double* h_b = (double*)malloc(num_matrices * n * sizeof(double));
    // double* h_x = (double*)malloc(num_matrices * n * sizeof(double));
    // double* h_A_copy = (double*)malloc(num_matrices * n * n * sizeof(double));
    int* h_P = (int*)malloc(num_matrices * n * sizeof(int));
    
    // Device allocations
    double *d_A, *d_b, *d_x;
    int *d_P;
    cudaMalloc(&d_A, num_matrices * n * n * sizeof(double));
    cudaMalloc(&d_P, num_matrices * n * sizeof(int));
    cudaMalloc(&d_b, num_matrices * n * sizeof(double));
    cudaMalloc(&d_x, num_matrices * n * sizeof(double));
    
    // Initialize matrices and vectors here...
    // (You would typically load your CSV data into h_A and h_b here)

    // double** A = read_matrix_from_csv("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/A_matrix_case1.csv", n, n);
    // double** A_copy = read_matrix_from_csv("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/A_matrix_case1.csv", n, n);
    // double** B_matrix = read_matrix_from_csv("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/B_matrix_case1", n, 1);
    // double** X_true_matrix = read_matrix_from_csv("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/Case_1_soln.csv", n, 1);


     double* h_A = read_matrix_from_csv_flat("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/A_matrix_case1.csv", n, n);
    //  printf("File got read");
     double* h_A_copy = read_matrix_from_csv_flat("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/A_matrix_case1.csv", n, n);
     double* h_b = read_matrix_from_csv_flat("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/B_matrix_case1.csv", n, 1);
     double* h_x = read_matrix_from_csv_flat("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/Case_1_soln.csv", n, 1);

    
    // Copy data to device
    cudaMemcpy(d_A, h_A, num_matrices * n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, num_matrices * n * sizeof(double), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (num_matrices + threads_per_block - 1) / threads_per_block;
    lu_solve_kernel<<<blocks, threads_per_block>>>(d_A, d_P, d_b, d_x, n);
    
    // Copy results back
    cudaMemcpy(h_x, d_x, num_matrices * n * sizeof(double), cudaMemcpyDeviceToHost);
    printf("Hii there");
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_P);
    cudaFree(d_b);
    cudaFree(d_x);
    free(h_A);
    free(h_b);
    free(h_x);
    free(h_P);
    
    return 0;
}