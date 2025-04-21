#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>
#include <float.h>

// ---------------------- Constants ----------------------
#define BLOCK_SIZE 16  // Common block size

// ---------------------- Error Handling ----------------------
#define CHECK_CUDA_ERROR(call) {                                 \
    cudaError_t err = call;                                      \
    if (err != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error in %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err));    \
        exit(EXIT_FAILURE);                                      \
    }                                                            \
}

// ---------------------- File IO Utilities (Unified Memory) ----------------------
// These functions allocate memory in unified memory (using cudaMallocManaged)
// so that all matrices and vectors reside in memory accessible both by CPU and GPU.

double* read_matrix_csv(const char* filename, int* rows, int* cols) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }
    char line[100000];
    *rows = 0;
    *cols = 0;
    while (fgets(line, sizeof(line), file)) {
        (*rows)++;
        if (*rows == 1) {
            char* token = strtok(line, ",\n");
            while (token) {
                (*cols)++;
                token = strtok(NULL, ",\n");
            }
        }
    }
    rewind(file);
    double* matrix;
    CHECK_CUDA_ERROR(cudaMallocManaged(&matrix, (*rows) * (*cols) * sizeof(double)));
    int row = 0;
    while (fgets(line, sizeof(line), file) && row < *rows) {
        char* token = strtok(line, ",\n");
        int col = 0;
        while (token && col < *cols) {
            matrix[row * (*cols) + col] = atof(token);
            token = strtok(NULL, ",\n");
            col++;
        }
        row++;
    }
    fclose(file);
    return matrix;
}

double* read_vector_csv(const char* filename, int* size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }
    char line[10000];
    *size = 0;
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",\n");
        while (token) {
            (*size)++;
            token = strtok(NULL, ",\n");
        }
    }
    rewind(file);
    double* vector;
    CHECK_CUDA_ERROR(cudaMallocManaged(&vector, (*size) * sizeof(double)));
    int idx = 0;
    while (fgets(line, sizeof(line), file) && idx < *size) {
        char* token = strtok(line, ",\n");
        while (token && idx < *size) {
            vector[idx] = atof(token);
            token = strtok(NULL, ",\n");
            idx++;
        }
    }
    fclose(file);
    return vector;
}

// ---------------------- CUDA Kernels ----------------------

// Finds pivot element for a column (one thread per row in the column range)
__global__ void find_pivot_kernel(double* U, int k, int n, double* max_vals, int* max_idxs) {
    __shared__ double local_max[BLOCK_SIZE];
    __shared__ int local_idx[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = k + blockIdx.x * blockDim.x + tid;
    local_max[tid] = -1.0;
    local_idx[tid] = -1;
    if (i < n) {
        double abs_val = fabs(U[i * n + k]);
        local_max[tid] = abs_val;
        local_idx[tid] = i;
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (local_max[tid] < local_max[tid + stride]) {
                local_max[tid] = local_max[tid + stride];
                local_idx[tid] = local_idx[tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        max_vals[blockIdx.x] = local_max[0];
        max_idxs[blockIdx.x] = local_idx[0];
    }
}

// Compute multipliers for column k
__global__ void compute_L_kernel(double* U, double* L, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    if (i < n) {
        double pivot = U[k * n + k];
        if (fabs(pivot) > 1e-15) {
            L[i * n + k] = U[i * n + k] / pivot;
        } else {
            L[i * n + k] = U[i * n + k] / 1e-15;
        }
    }
}

// Swap rows in matrix
__global__ void swap_rows_kernel(double* matrix, int row1, int row2, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        double temp = matrix[row1 * n + j];
        matrix[row1 * n + j] = matrix[row2 * n + j];
        matrix[row2 * n + j] = temp;
    }
}

// NEW: Update kernel using one thread per element in the submatrix
__global__ void update_U_kernel_element(double* U, double* L, int k, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + k + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    if (i < n && j < n) {
        double l_ik = L[i * n + k];
        double u_kj = U[k * n + j];
        // Each thread subtracts its contribution
        U[i * n + j] -= l_ik * u_kj;
    }
}

// ---------------------- LU Decomposition (CUDA Version) ----------------------
// Performs LU decomposition with partial pivoting on unified memory data.
void lu_decomposition_gpu(double* A, double* L, double* U, int* P, int n) {
    double *d_max_vals;
    int *d_max_idxs;
    int pivotArraySize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CHECK_CUDA_ERROR(cudaMalloc(&d_max_vals, pivotArraySize * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_max_idxs, pivotArraySize * sizeof(int)));
    
    double *h_max_vals = (double*)malloc(pivotArraySize * sizeof(double));
    int *h_max_idxs = (int*)malloc(pivotArraySize * sizeof(int));
    
    // Initialize P, L and U
    for (int i = 0; i < n; i++) {
        P[i] = i;
        for (int j = 0; j < n; j++) {
            L[i * n + j] = (i == j) ? 1.0 : 0.0;
            U[i * n + j] = A[i * n + j];
        }
    }
    
    // Loop over each column k for elimination.
    for (int k = 0; k < n - 1; k++) {
        int num_rows = n - k;
        int grid_size = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        find_pivot_kernel<<<grid_size, BLOCK_SIZE>>>(U, k, n, d_max_vals, d_max_idxs);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        CHECK_CUDA_ERROR(cudaMemcpy(h_max_vals, d_max_vals, grid_size * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_max_idxs, d_max_idxs, grid_size * sizeof(int), cudaMemcpyDeviceToHost));
        
        double max_val = h_max_vals[0];
        int max_idx = h_max_idxs[0];
        for (int i = 1; i < grid_size; i++) {
            if (h_max_vals[i] > max_val) {
                max_val = h_max_vals[i];
                max_idx = h_max_idxs[i];
            }
        }
        
        if (max_idx < k || max_idx >= n || max_val < DBL_EPSILON) {
            continue;
        }
        
        if (max_idx != k) {
            swap_rows_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(U, k, max_idx, n);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            if (k > 0) {
                dim3 block_dim(BLOCK_SIZE);
                dim3 grid_dim((k + BLOCK_SIZE - 1) / BLOCK_SIZE);
                swap_rows_kernel<<<grid_dim, block_dim>>>(L, k, max_idx, n);
                CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            }
            int temp = P[k];
            P[k] = P[max_idx];
            P[max_idx] = temp;
        }
        
        grid_size = (n - k - 1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_L_kernel<<<grid_size, BLOCK_SIZE>>>(U, L, k, n);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Launch the updated update kernel: one thread per element.
        int submatrix_size = n - k - 1;
        dim3 block_dim(16, 16);
        dim3 grid_dim((submatrix_size + block_dim.x - 1) / block_dim.x,
                      (submatrix_size + block_dim.y - 1) / block_dim.y);
        update_U_kernel_element<<<grid_dim, block_dim>>>(U, L, k, n);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    
    cudaFree(d_max_vals);
    cudaFree(d_max_idxs);
    free(h_max_vals);
    free(h_max_idxs);
}

// ---------------------- Error Metrics and Matrix-Vector Multiplication ----------------------


void matrix_vector_multiply(double* A, double* x, double* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = 0.0;
        for (int j = 0; j < n; j++) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}


// ---------------------- Test Case A ----------------------
// Reads CSV files, performs LU decomposition, solves the system, and computes errors.
void run_test_case_A(const char* matrix_file, const char* vector_file, const char* solution_file) {
    printf("\n----- Running Test Case A (Sequential) -----\n\n");
    
    int rows, cols;
    double* A = read_matrix_csv(matrix_file, &rows, &cols);
    if (!A || rows != cols) {
        printf("Error: Could not read matrix A or matrix is not square\n");
        if (A) free(A);
        return;
    }
    int n = rows;
    printf("Successfully read matrix A of size %d x %d\n", n, n);
    
    int b_size;
    double* b = read_vector_csv(vector_file, &b_size);
    if (!b || b_size != n) {
        printf("Error: Could not read vector b or size mismatch\n");
        free(A);
        if (b) free(b);
        return;
    }
    printf("Successfully read vector b of size %d\n", b_size);
    
    int x_true_size;
    double* x_true = read_vector_csv(solution_file, &x_true_size);
    if (!x_true || x_true_size != n) {
        printf("Error: Could not read true solution or size mismatch\n");
        free(A);
        free(b);
        if (x_true) free(x_true);
        return;
    }
    printf("Successfully read true solution of size %d\n", x_true_size);
    
    double *L = (double*)malloc(n * n * sizeof(double));
    double *U = (double*)malloc(n * n * sizeof(double));
    int* P = (int*)malloc(n * sizeof(int));
    
    clock_t lu_start = clock();
    lu_decomposition_seq(A, L, U, P, n);
    clock_t lu_end = clock();
    double lu_time = ((double)(lu_end - lu_start)) / CLOCKS_PER_SEC;
    printf("LU Decomposition Time: %f seconds\n", lu_time);
    
    clock_t err_start = clock();
    double E1 = calculate_error_E1_seq(A, L, U, P, n);
    clock_t err_end = clock();
    double err_time = ((double)(err_end - err_start)) / CLOCKS_PER_SEC;
    printf("Error Metric Calculation Time: %f seconds\n", err_time);
    printf("Error Metric 1 (E1): %.2e\n", E1);
    
    double* ones = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        ones[i] = 1.0;
    }
    
    double* row_sums = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        row_sums[i] = 0.0;
        for (int j = 0; j < n; j++) {
            row_sums[i] += A[i * n + j];
        }
    }
    
    double* x_ones = (double*)malloc(n * sizeof(double));
    solve_system_seq(A, L, U, P, row_sums, x_ones, n, 3);
    double E2 = calculate_error_E3_seq(x_ones, ones, n);
    printf("Error Metric 2 (E2): %.2e\n", E2);
    
    double* x_solved = (double*)malloc(n * sizeof(double));
    solve_system_seq(A, L, U, P, b, x_solved, n, 3);
    double E3 = calculate_error_E3_seq(x_solved, x_true, n);
    printf("Error Metric 3 (E3): %.2e\n", E3);
    
    printf("Total Computation Time (LU + Error): %f seconds\n", lu_time + err_time);
    
    free(A);
    free(b);
    free(x_true);
    free(ones);
    free(row_sums);
    free(x_ones);
    free(x_solved);
    free(L);
    free(U);
    free(P);
    
    printf("\n----- Test Case A Completed -----\n");
}

// ---------------------- Main Function ----------------------
// Loops through matrix sizes and runs the sequential test for each.
int main() {
    printf("Running Sequential LU Decomposition Implementation\n");
    
    int sizes[] = {32, 64, 128, 256, 512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    char matrix_filename[256];
    char b_filename[256];
    char sol_filename[256];
    
    for (int idx = 0; idx < num_sizes; idx++) {
        int size = sizes[idx];
        printf("\n=== Running Test Case for Matrix Size %d x %d ===\n", size, size);
        
        sprintf(matrix_filename, "/content/generated_matrices/A_%d.csv", size);
        sprintf(b_filename, "/content/generated_matrices/B_%d.csv", size);
        sprintf(sol_filename, "/content/generated_matrices/x_true_%d.csv", size);
        
        run_test_case_A(matrix_filename, b_filename, sol_filename);
    }
    
    return 0;
}
