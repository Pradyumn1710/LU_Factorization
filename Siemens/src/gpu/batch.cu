#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Configuration
const int MATRIX_SIZE = 100;    // Size of each square matrix (n x n)
const int NUM_MATRICES = 500;    // Number of matrices in batch
const int THREADS_PER_BLOCK = 32; // Threads per block (power of 2 for reductions)

// Kernel for parallel LU decomposition with partial pivoting
__global__ void lu_decomposition_kernel(double *A, long *P, int num_matrices, int n) {
    const int matrix_id = blockIdx.x;
    if (matrix_id >= num_matrices) return;

    double *matrix = A + matrix_id * n * n;
    long *perm = P + matrix_id * n;

    // Initialize permutation vector
    if (threadIdx.x < n) {
        perm[threadIdx.x] = threadIdx.x;
    }
    __syncthreads();

    for (int k = 0; k < n - 1; ++k) {
        // Parallel pivot selection
        int max_row = k;
        double max_val = fabs(matrix[k * n + k]);

        // Each thread checks a subset of rows
        for (int i = k + threadIdx.x; i < n; i += blockDim.x) {
            double current = fabs(matrix[i * n + k]);
            if (current > max_val || (current == max_val && i > max_row)) {
                max_val = current;
                max_row = i;
            }
        }

        // Shared memory for reduction
        __shared__ int s_max_row[THREADS_PER_BLOCK];
        __shared__ double s_max_val[THREADS_PER_BLOCK];
        s_max_row[threadIdx.x] = max_row;
        s_max_val[threadIdx.x] = max_val;
        __syncthreads();

        // Parallel reduction to find global max
        for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                if (s_max_val[threadIdx.x + stride] > s_max_val[threadIdx.x] ||
                   (s_max_val[threadIdx.x + stride] == s_max_val[threadIdx.x] &&
                    s_max_row[threadIdx.x + stride] > s_max_row[threadIdx.x])) {
                    s_max_val[threadIdx.x] = s_max_val[threadIdx.x + stride];
                    s_max_row[threadIdx.x] = s_max_row[threadIdx.x + stride];
                }
            }
            __syncthreads();
        }

        // Broadcast results
        __shared__ int global_max_row;
        __shared__ double global_max_val;
        if (threadIdx.x == 0) {
            global_max_row = s_max_row[0];
            global_max_val = s_max_val[0];
        }
        __syncthreads();

        // Parallel row swapping
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            double temp = matrix[k * n + j];
            matrix[k * n + j] = matrix[global_max_row * n + j];
            matrix[global_max_row * n + j] = temp;
        }

        // Update permutation vector
        if (threadIdx.x == 0 && global_max_row != k) {
            long temp = perm[k];
            perm[k] = perm[global_max_row];
            perm[global_max_row] = temp;
        }
        __syncthreads();

        // Compute multipliers
        const double pivot = matrix[k * n + k];
        for (int i = k + 1 + threadIdx.x; i < n; i += blockDim.x) {
            matrix[i * n + k] /= pivot;
        }
        __syncthreads();

        // Update submatrix using coalesced accesses
        for (int j = k + 1 + threadIdx.x; j < n; j += blockDim.x) {
            const double pivot_element = matrix[k * n + j];
            for (int i = k + 1; i < n; ++i) {
                matrix[i * n + j] -= matrix[i * n + k] * pivot_element;
            }
        }
        __syncthreads();
    }
}

// Helper function to read matrix from file
double* read_matrix(const char* filename, int rows, int cols) {
    FILE* file = fopen(filename, "r");
    double* matrix = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows * cols; ++i) {
        fscanf(file, "%lf,", &matrix[i]);
    }
    fclose(file);
    return matrix;
}

int main() {
    // Host allocations
    double *h_A = read_matrix("/home/pradyumn/Academic/Non_college/Main/Siemens/data/mytests/matrix_100x100.csv", MATRIX_SIZE, MATRIX_SIZE);
    double *d_A;
    long *d_P;

    // Device allocations
    CHECK_CUDA(cudaMalloc(&d_A, NUM_MATRICES * MATRIX_SIZE * MATRIX_SIZE * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_P, NUM_MATRICES * MATRIX_SIZE * sizeof(long)));

    // Copy matrices to device (using same matrix for all tests)
    for (int i = 0; i < NUM_MATRICES; ++i) {
        CHECK_CUDA(cudaMemcpy(d_A + i * MATRIX_SIZE * MATRIX_SIZE, h_A,
                            MATRIX_SIZE * MATRIX_SIZE * sizeof(double),
                            cudaMemcpyHostToDevice));
    }

    // Kernel configuration
    dim3 grid(NUM_MATRICES);
    dim3 block(THREADS_PER_BLOCK);

    // Warmup
    for (int i = 0; i < 3; ++i) {
        lu_decomposition_kernel<<<grid, block>>>(d_A, d_P, NUM_MATRICES, MATRIX_SIZE);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    lu_decomposition_kernel<<<grid, block>>>(d_A, d_P, NUM_MATRICES, MATRIX_SIZE);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("Batched LU Decomposition (%d matrices):\n", NUM_MATRICES);
    printf("Total time: %.3f ms | Per matrix: %.3f ms\n", ms, ms/NUM_MATRICES);

    // Cleanup
    free(h_A);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_P));

    return 0;
}