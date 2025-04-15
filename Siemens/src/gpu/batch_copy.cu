#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
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

const int MATRIX_SIZE = 2048;
const int BATCH_SIZE = 200;
const int WARPS_PER_BLOCK = 8;
const int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
const int PIVOT_WARPS = 4;
const int TILE_SIZE = 64;

__device__ __forceinline__ void warp_reduce(double& val, int& idx) {
    using namespace cooperative_groups;
    thread_block_tile<32> tile = tiled_partition<32>(this_thread_block());
    
    for(int i = tile.size()/2; i > 0; i >>= 1) {
        double tmp_val = tile.shfl_down(val, i);
        int tmp_idx = tile.shfl_down(idx, i);
        if(tmp_val > val || (tmp_val == val && tmp_idx > idx)) {
            val = tmp_val;
            idx = tmp_idx;
        }
    }
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
gpu_lu_optimized(double* A, int* P, int n, int batch_size) {
    extern __shared__ __align__(sizeof(double)) char shared_mem[];
    double* smem = reinterpret_cast<double*>(shared_mem);
    int* s_pivot = reinterpret_cast<int*>(shared_mem + n*n*sizeof(double));
    
    const int matrix_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    if(matrix_id >= batch_size) return;
    
    double* matrix = A + matrix_id * n * n;
    int* perm = P + matrix_id * n;

    // Vectorized global to shared memory load
    const int vec_size = sizeof(float4)/sizeof(double);
    const int tiles = (n * n) / (blockDim.x * vec_size);
    float4* smem_vec = reinterpret_cast<float4*>(smem);
    const float4* gmem_vec = reinterpret_cast<float4*>(matrix);

    for(int i = tid; i < tiles; i += blockDim.x) {
        smem_vec[i] = gmem_vec[i];
    }
    
    // Initialize permutation vector
    if(tid < n) s_pivot[tid] = tid;
    __syncthreads();

    for(int k = 0; k < n; ++k) {
        // Parallel pivot search with warp shuffles
        double max_val = fabs(smem[k*n + k]);
        int max_row = k;

        for(int i = k + warp_id; i < n; i += WARPS_PER_BLOCK) {
            double current = fabs(smem[i*n + k]);
            if(current > max_val || (current == max_val && i > max_row)) {
                max_val = current;
                max_row = i;
            }
        }

        warp_reduce(max_val, max_row);
        
        __shared__ double s_max_vals[WARPS_PER_BLOCK];
        __shared__ int s_max_rows[WARPS_PER_BLOCK];
        
        if(lane_id == 0) {
            s_max_vals[warp_id] = max_val;
            s_max_rows[warp_id] = max_row;
        }
        __syncthreads();

        if(warp_id == 0) {
            double final_val = s_max_vals[lane_id];
            int final_row = s_max_rows[lane_id];
            if(lane_id < WARPS_PER_BLOCK) {
                for(int i = lane_id + WARPS_PER_BLOCK/2; i < WARPS_PER_BLOCK; i += WARPS_PER_BLOCK/2) {
                    if(s_max_vals[i] > final_val || (s_max_vals[i] == final_val && s_max_rows[i] > final_row)) {
                        final_val = s_max_vals[i];
                        final_row = s_max_rows[i];
                    }
                }
            }
            if(tid == 0) {
                s_max_vals[0] = final_val;
                s_max_rows[0] = final_row;
            }
        }
        __syncthreads();

        // Row swapping with coalesced access
        if(s_max_rows[0] != k) {
            for(int j = tid; j < n; j += blockDim.x) {
                double temp = smem[k*n + j];
                smem[k*n + j] = smem[s_max_rows[0]*n + j];
                smem[s_max_rows[0]*n + j] = temp;
            }
            if(tid == 0) {
                int temp = s_pivot[k];
                s_pivot[k] = s_pivot[s_max_rows[0]];
                s_pivot[s_max_rows[0]] = temp;
            }
        }
        __syncthreads();

        // Tiled LU decomposition
        const double pivot = smem[k*n + k];
        const int tiles = (n - k - 1 + TILE_SIZE - 1) / TILE_SIZE;

        for(int tile = 0; tile < tiles; ++tile) {
            const int row_start = k + 1 + tile * TILE_SIZE;
            const int row_end = min(row_start + TILE_SIZE, n);
            
            // Compute multipliers
            for(int i = row_start + tid; i < row_end; i += blockDim.x) {
                smem[i*n + k] /= pivot;
            }
            __syncthreads();

            // Update submatrix
            for(int j = k + 1 + tid; j < n; j += blockDim.x) {
                const double pivot_element = smem[k*n + j];
                #pragma unroll
                for(int i = row_start; i < row_end; ++i) {
                    smem[i*n + j] -= smem[i*n + k] * pivot_element;
                }
            }
            __syncthreads();
        }
    }

    // Vectorized shared to global memory store
    float4* gmem_out_vec = reinterpret_cast<float4*>(matrix);
    for(int i = tid; i < tiles; i += blockDim.x) {
        gmem_out_vec[i] = smem_vec[i];
    }
    
    // Store permutation vector
    if(tid < n) perm[tid] = s_pivot[tid];
}

int main() {
    const size_t matrix_mem = MATRIX_SIZE * MATRIX_SIZE * sizeof(double);
    const size_t perm_mem = MATRIX_SIZE * sizeof(int);
    
    // Host allocations with pinned memory
    double *h_A, *h_A_d;
    int *h_P;
    CHECK_CUDA(cudaMallocHost(&h_A, BATCH_SIZE * matrix_mem));
    CHECK_CUDA(cudaMallocHost(&h_A_d, BATCH_SIZE * matrix_mem));
    CHECK_CUDA(cudaMallocHost(&h_P, BATCH_SIZE * perm_mem));

    // Initialize matrices with OpenMP parallelization
    #pragma omp parallel for num_threads(16)
    for(int m = 0; m < BATCH_SIZE; ++m) {
        for(int i = 0; i < MATRIX_SIZE; ++i) {
            for(int j = 0; j < MATRIX_SIZE; ++j) {
                h_A[m*MATRIX_SIZE*MATRIX_SIZE + i*MATRIX_SIZE + j] = 
                    (double)rand() / RAND_MAX + 1.0;
            }
        }
    }

    // Device allocations with async streams
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    double *d_A;
    int *d_P;
    CHECK_CUDA(cudaMallocAsync(&d_A, BATCH_SIZE * matrix_mem, stream));
    CHECK_CUDA(cudaMallocAsync(&d_P, BATCH_SIZE * perm_mem, stream));
    
    // Async H2D copy
    CHECK_CUDA(cudaMemcpyAsync(d_A, h_A, BATCH_SIZE * matrix_mem, 
                              cudaMemcpyHostToDevice, stream));

    // Kernel configuration
    const size_t shared_mem = MATRIX_SIZE*MATRIX_SIZE*sizeof(double) + 
                             MATRIX_SIZE*sizeof(int);
    cudaFuncSetAttribute(gpu_lu_optimized, 
                        cudaFuncAttributeMaxDynamicSharedMemorySize, 
                        shared_mem);
    
    // Warmup
    for(int i = 0; i < 3; ++i) {
        gpu_lu_optimized<<<BATCH_SIZE, THREADS_PER_BLOCK, shared_mem, stream>>>(
            d_A, d_P, MATRIX_SIZE, BATCH_SIZE);
    }
    
    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start, stream));
    gpu_lu_optimized<<<BATCH_SIZE, THREADS_PER_BLOCK, shared_mem, stream>>>(
        d_A, d_P, MATRIX_SIZE, BATCH_SIZE);
    CHECK_CUDA(cudaEventRecord(stop, stream));
    
    // Async D2H copy
    CHECK_CUDA(cudaMemcpyAsync(h_A_d, d_A, BATCH_SIZE * matrix_mem,
                              cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(h_P, d_P, BATCH_SIZE * perm_mem,
                              cudaMemcpyDeviceToHost, stream));
    
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("Optimized LU Decomposition Performance:\n");
    printf("Matrix Size: %d | Batch Size: %d\n", MATRIX_SIZE, BATCH_SIZE);
    printf("Total Time: %.3fms | Per Matrix: %.6fms\n", ms, ms/BATCH_SIZE);

    // Cleanup
    CHECK_CUDA(cudaFreeAsync(d_A, stream));
    CHECK_CUDA(cudaFreeAsync(d_P, stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_A_d));
    CHECK_CUDA(cudaFreeHost(h_P));

    return 0;
}