#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <omp.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

const int MATRIX_SIZES[] = {2048};
const int BATCH_SIZES[] = {200};
const int WARMUP_RUNS = 3;
const int BENCHMARK_RUNS = 10;

void lu_decomposition(double* A, double* L, double* U, int* P, int n) {
    for(int i = 0; i < n; i++) {
        P[i] = i;
        for(int j = 0; j < n; j++) {
            L[i*n + j] = (i == j) ? 1.0 : 0.0;
            U[i*n + j] = A[i*n + j];
        }
    }

    for(int k = 0; k < n; k++) {
        int max_row = k;
        double max_val = fabs(U[k*n + k]);
        for(int i = k+1; i < n; i++) {
            if(fabs(U[i*n + k]) > max_val) {
                max_val = fabs(U[i*n + k]);
                max_row = i;
            }
        }

        if(max_row != k) {
            for(int j = 0; j < n; j++) {
                double temp = U[k*n + j];
                U[k*n + j] = U[max_row*n + j];
                U[max_row*n + j] = temp;
                
                if(j < k) {
                    temp = L[k*n + j];
                    L[k*n + j] = L[max_row*n + j];
                    L[max_row*n + j] = temp;
                }
            }
            int temp = P[k];
            P[k] = P[max_row];
            P[max_row] = temp;
        }

        for(int i = k+1; i < n; i++) {
            L[i*n + k] = U[i*n + k] / U[k*n + k];
            for(int j = k; j < n; j++) {
                U[i*n + j] -= L[i*n + k] * U[k*n + j];
            }
        }
    }
}

__global__ void gpu_lu_kernel(double* A, double* L, double* U, int* P, int n, int batch_size) {
    const int matrix_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int matrix_offset = matrix_id * n * n;
    const int perm_offset = matrix_id * n;
    
    extern __shared__ double shared_mem[];
    double* smem = shared_mem;
    int* s_pivot = (int*)&shared_mem[n*n];

    if(matrix_id >= batch_size) return;

    for(int i = tid; i < n; i += blockDim.x) {
        for(int j = 0; j < n; j++) {
            smem[i*n + j] = A[matrix_offset + i*n + j];
        }
        s_pivot[i] = i;
    }
    __syncthreads();

    for(int k = 0; k < n; k++) {
        double max_val = fabs(smem[k*n + k]);
        int max_row = k;
        
        for(int i = k + tid; i < n; i += blockDim.x) {
            double current = fabs(smem[i*n + k]);
            if(current > max_val) {
                max_val = current;
                max_row = i;
            }
        }
        
        __shared__ double s_max_val[256];
        __shared__ int s_max_row[256];
        s_max_val[tid] = max_val;
        s_max_row[tid] = max_row;
        
        for(int stride = blockDim.x/2; stride > 0; stride >>= 1) {
            __syncthreads();
            if(tid < stride && s_max_val[tid + stride] > s_max_val[tid]) {
                s_max_val[tid] = s_max_val[tid + stride];
                s_max_row[tid] = s_max_row[tid + stride];
            }
        }

        if(tid == 0 && s_max_row[0] != k) {
            for(int j = 0; j < n; j++) {
                double temp = smem[k*n + j];
                smem[k*n + j] = smem[s_max_row[0]*n + j];
                smem[s_max_row[0]*n + j] = temp;
            }
            int temp = s_pivot[k];
            s_pivot[k] = s_pivot[s_max_row[0]];
            s_pivot[s_max_row[0]] = temp;
        }
        __syncthreads();

        const double diag = smem[k*n + k];
        for(int i = k + 1 + tid; i < n; i += blockDim.x) {
            const double factor = smem[i*n + k] / diag;
            smem[i*n + k] = factor;
            for(int j = k; j < n; j++) {
                smem[i*n + j] -= factor * smem[k*n + j];
            }
        }
        __syncthreads();
    }

    for(int i = tid; i < n; i += blockDim.x) {
        for(int j = 0; j < n; j++) {
            L[matrix_offset + i*n + j] = (i > j) ? smem[i*n + j] : (i == j) ? 1.0 : 0.0;
            U[matrix_offset + i*n + j] = (i <= j) ? smem[i*n + j] : 0.0;
        }
        P[perm_offset + i] = s_pivot[i];
    }
}

int main() {
    FILE* csv_file = fopen("results.csv", "w");
    fprintf(csv_file, "MatrixSize,BatchSize,CPUTime(s),GPUTime(ms),Speedup\n");

    for(int size_idx = 0; size_idx < sizeof(MATRIX_SIZES)/sizeof(int); size_idx++) {
        int n = MATRIX_SIZES[size_idx];
        
        for(int batch_idx = 0; batch_idx < sizeof(BATCH_SIZES)/sizeof(int); batch_idx++) {
            int batch_size = BATCH_SIZES[batch_idx];
            size_t matrix_mem = n * n * sizeof(double);
            
            // Host memory
            double *h_A = (double*)malloc(batch_size * matrix_mem);
            double *h_L_cpu = (double*)malloc(batch_size * matrix_mem);
            double *h_U_cpu = (double*)malloc(batch_size * matrix_mem);
            int *h_P_cpu = (int*)malloc(batch_size * n * sizeof(int));

            // Initialize matrices
            #pragma omp parallel for
            for(int m = 0; m < batch_size; m++) {
                for(int i = 0; i < n; i++) {
                    for(int j = 0; j < n; j++) {
                        h_A[m*n*n + i*n + j] = (double)rand() / RAND_MAX;
                    }
                }
            }

            // Time CPU
            double cpu_start = omp_get_wtime();
            #pragma omp parallel for num_threads(8)
            for(int m = 0; m < batch_size; m++) {
                lu_decomposition(h_A + m*n*n, h_L_cpu + m*n*n, h_U_cpu + m*n*n, h_P_cpu + m*n, n);
            }
            double cpu_time = omp_get_wtime() - cpu_start;

            // GPU setup
            double *d_A, *d_L, *d_U;
            int *d_P;
            CHECK_CUDA(cudaMalloc(&d_A, batch_size * matrix_mem));
            CHECK_CUDA(cudaMalloc(&d_L, batch_size * matrix_mem));
            CHECK_CUDA(cudaMalloc(&d_U, batch_size * matrix_mem));
            CHECK_CUDA(cudaMalloc(&d_P, batch_size * n * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_A, h_A, batch_size * matrix_mem, cudaMemcpyHostToDevice));

            // Warmup
            for(int i = 0; i < WARMUP_RUNS; i++) {
                gpu_lu_kernel<<<batch_size, 256, n*n*sizeof(double) + n*sizeof(int)>>>(d_A, d_L, d_U, d_P, n, batch_size);
            }
            CHECK_CUDA(cudaDeviceSynchronize());

            // Benchmark
            cudaEvent_t start, stop;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));
            float total_ms = 0;
            
            for(int i = 0; i < BENCHMARK_RUNS; i++) {
                CHECK_CUDA(cudaEventRecord(start));
                gpu_lu_kernel<<<batch_size, 256, n*n*sizeof(double) + n*sizeof(int)>>>(d_A, d_L, d_U, d_P, n, batch_size);
                CHECK_CUDA(cudaEventRecord(stop));
                CHECK_CUDA(cudaEventSynchronize(stop));
                
                float ms;
                CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
                total_ms += ms;
            }
            double avg_gpu_ms = total_ms / BENCHMARK_RUNS;

            // Write to CSV
            fprintf(csv_file, "%d,%d,%.3f,%.6f,%.2f\n",
                    n, batch_size, cpu_time, avg_gpu_ms, 
                    cpu_time / (avg_gpu_ms / 1000.0));
            fflush(csv_file);

            // Cleanup
            free(h_A);
            free(h_L_cpu);
            free(h_U_cpu);
            free(h_P_cpu);
            CHECK_CUDA(cudaFree(d_A));
            CHECK_CUDA(cudaFree(d_L));
            CHECK_CUDA(cudaFree(d_U));
            CHECK_CUDA(cudaFree(d_P));
        }
    }

    fclose(csv_file);
    printf("Results saved to results.csv\n");
    return 0;
}