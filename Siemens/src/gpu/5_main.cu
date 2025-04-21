#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <math.h>

#define IDX(i, j, N) ((i)*(N)+(j))

__device__ void swap_rows(float* A, int row1, int row2, int size) {
    for (int j = 0; j < size; ++j) {
        float temp = A[IDX(row1, j, size)];
        A[IDX(row1, j, size)] = A[IDX(row2, j, size)];
        A[IDX(row2, j, size)] = temp;
    }
}

__device__ unsigned int lu_decompose_global(float* A, int size) {
    unsigned int start = clock();
    for (int k = 0; k < size - 1; ++k) {
        // Partial pivoting: find row with max absolute value in column k
        int max_row = k;
        float max_val = fabsf(A[IDX(k, k, size)]);
        for (int i = k + 1; i < size; ++i) {
            float val = fabsf(A[IDX(i, k, size)]);
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }

        if (max_val == 0.0f) {
            printf("⚠️ Zero column at k = %d\n", k);
            return 0;
        }

        if (max_row != k) {
            swap_rows(A, k, max_row, size);
        }

        for (int i = k + 1; i < size; ++i) {
            A[IDX(i, k, size)] /= A[IDX(k, k, size)];
            for (int j = k + 1; j < size; ++j) {
                A[IDX(i, j, size)] -= A[IDX(i, k, size)] * A[IDX(k, j, size)];
            }
        }
    }
    return clock() - start;
}

__global__ void lu_kernel_direct(float* global_matrices, int size, unsigned int* lu_times) {
    int matrix_id = blockIdx.x;

    if (threadIdx.x == 0) {
        float* A = &global_matrices[matrix_id * size * size];

        // printf("🚀 Starting LU for matrix %d\n", matrix_id);
        unsigned int cycles = lu_decompose_global(A, size);
        // printf("✅ LU done for matrix %d, cycles = %u\n", matrix_id, cycles);

        lu_times[matrix_id] = cycles;
    }
}

void read_csv(const char* filename, float* matrix, int size) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("❌ Error opening file: %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < size * size; ++i) {
        if (fscanf(fp, "%f,", &matrix[i]) != 1) {
            printf("❌ Error reading index %d\n", i);
            exit(1);
        }
    }
    fclose(fp);
}

void replicate_matrix(const float* src, float* dest, int size, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < size * size; ++j) {
            dest[i * size * size + j] = src[j];
        }
    }
}

int main() {
    // /home/pradyumn/Academic/Non_college/Main/Siemens/data/mytests/matrix_1000x1000.csv
    const char* filename = "/home/pradyumn/Academic/Non_college/Main/Siemens/data/mytests/matrix_1000x1000.csv";
    int size = 1000;
    int num_matrices = 500;
    int matrix_elems = size * size;
    size_t matrix_bytes = matrix_elems * sizeof(float);
    size_t total_bytes = matrix_bytes * num_matrices;

    float* h_single = (float*)malloc(matrix_bytes);
    float* h_all = (float*)malloc(total_bytes);
    read_csv(filename, h_single, size);
    replicate_matrix(h_single, h_all, size, num_matrices);

    float* d_all;
    unsigned int* d_lu_times;
    unsigned int* h_lu_times = (unsigned int*)malloc(num_matrices * sizeof(unsigned int));
    cudaMalloc(&d_all, total_bytes);
    cudaMemcpy(d_all, h_all, total_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_lu_times, num_matrices * sizeof(unsigned int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    lu_kernel_direct<<<num_matrices,128>>>(d_all, size, d_lu_times);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    cudaMemcpy(h_lu_times, d_lu_times, num_matrices * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("⏱️ Kernel execution time: %.3f ms\n", kernel_ms);
    unsigned long long total_cycles = 0;
    for (int i = 0; i < num_matrices; ++i) {
        // printf("🔍 Matrix %d LU cycles = %u\n", i, h_lu_times[i]);
        total_cycles += h_lu_times[i];
    }
    printf("🔧 Avg LU cycles: %llu\n", total_cycles / num_matrices);

    cudaFree(d_all);
    cudaFree(d_lu_times);
    free(h_single);
    free(h_all);
    free(h_lu_times);

    return 0;
}