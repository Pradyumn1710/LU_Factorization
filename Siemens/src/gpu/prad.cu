#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

// Configuration - Adjusted for 6GB VRAM
#define NUM_MATRICES      750     // 750 * 1000x1000 * 8B = 5.96GB
#define MATRIX_SIZE       1000
#define THREADS_PER_BLOCK 256     // Optimal for task parallelism
#define CHECK_CUDA(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %d: %s (%s:%d)\n", \
                err_, cudaGetErrorString(err_), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Read column-major CSV (1 column per line)
double** read_csv_column_major(const char* filename, int size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("File open failed");
        exit(EXIT_FAILURE);
    }

    double** matrix = (double**)malloc(size * sizeof(double*));
    for (int col = 0; col < size; col++) {
        matrix[col] = (double*)malloc(size * sizeof(double));
        for (int row = 0; row < size; row++) {
            if (fscanf(file, "%lf", &matrix[col][row]) != 1) {
                fprintf(stderr, "Read error at column %d, row %d\n", col, row);
                exit(EXIT_FAILURE);
            }
            if (row < size - 1) fgetc(file); // Skip comma
        }
    }
    fclose(file);
    return matrix;
}

__device__ void lu_decomposition(double* mat, int n) {
    for (int k = 0; k < n; k++) {
        // Pivot finding
        int max_row = k;
        double max_val = fabs(mat[k*n + k]);
        for (int i = k+1; i < n; i++) {
            double val = fabs(mat[i*n + k]);
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }

        // Row swap
        if (max_row != k) {
            for (int j = 0; j < n; j++) {
                double temp = mat[k*n + j];
                mat[k*n + j] = mat[max_row*n + j];
                mat[max_row*n + j] = temp;
            }
        }

        // Factorization
        double diag = mat[k*n + k];
        for (int i = k+1; i < n; i++) {
            double factor = mat[i*n + k] / diag;
            mat[i*n + k] = factor;
            
            for (int j = k+1; j < n; j++) {
                mat[i*n + j] -= factor * mat[k*n + j];
            }
        }
    }
}

__global__ void lu_kernel(double* matrices, int size) {
    const int mat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (mat_idx >= NUM_MATRICES) return;
    
    double* A = matrices + mat_idx * size * size;
    lu_decomposition(A, size);
}

int main() {
    const size_t single_matrix_size = MATRIX_SIZE * MATRIX_SIZE * sizeof(double);
    const size_t total_device_mem = NUM_MATRICES * single_matrix_size;

    // 1. Read input matrix (column-major)
    double** host_column_major = read_csv_column_major(
        "/home/pradyumn/Academic/Non_college/Main/Siemens/data/mytests/matrix_1000x1000.csv", MATRIX_SIZE
    );

    // 2. Allocate pinned memory for row-major matrices
    double* h_matrices;
    CHECK_CUDA(cudaMallocHost(&h_matrices, total_device_mem));

    // 3. Convert to row-major with transposition (parallelized)
    #pragma omp parallel for collapse(3)
    for (int m = 0; m < NUM_MATRICES; m++) {
        for (int row = 0; row < MATRIX_SIZE; row++) {
            for (int col = 0; col < MATRIX_SIZE; col++) {
                const int dst_idx = m*MATRIX_SIZE*MATRIX_SIZE + row*MATRIX_SIZE + col;
                const int src_idx = col*MATRIX_SIZE + row;  // Transpose
                h_matrices[dst_idx] = host_column_major[col][row];
            }
        }
    }

    // 4. Device memory management
    double* d_matrices;
    CHECK_CUDA(cudaMalloc(&d_matrices, total_device_mem));
    CHECK_CUDA(cudaMemcpy(d_matrices, h_matrices, total_device_mem, 
                        cudaMemcpyHostToDevice));

    // 5. Kernel configuration
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((NUM_MATRICES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // 6. Warm-up kernel
    lu_kernel<<<grid, block>>>(d_matrices, MATRIX_SIZE);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 7. Timed execution
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    lu_kernel<<<grid, block>>>(d_matrices, MATRIX_SIZE);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("Execution Time: %.3f seconds\n", ms / 1000.0f);

    // 8. Cleanup
    CHECK_CUDA(cudaFree(d_matrices));
    CHECK_CUDA(cudaFreeHost(h_matrices));
    for (int i = 0; i < MATRIX_SIZE; i++) free(host_column_major[i]);
    free(host_column_major);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}