#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define NUM_MATRICES 10000
#define N 500
#define CACHE_LINE_SIZE 128  // Typical cache line size

typedef struct {
    double data[N][N] __attribute__((aligned(CACHE_LINE_SIZE)));
    int permutation[N];
} Matrix;

double** read_matrix_from_csv_(const char* filename, int rows, int cols) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    double** matrix = (double**)aligned_alloc(CACHE_LINE_SIZE, rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)aligned_alloc(CACHE_LINE_SIZE, cols * sizeof(double));
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%lf", &matrix[i][j]) != 1) {
                fprintf(stderr, "Error reading matrix element at (%d,%d)\n", i, j);
                exit(EXIT_FAILURE);
            }
            if (j < cols - 1) fgetc(file);
        }
    }
    fclose(file);
    return matrix;
}

void lu_decomposition_single(Matrix* matrix) {
    double (*A)[N] = matrix->data;
    int* P = matrix->permutation;
    
    // Initialize permutation vector
    for (int i = 0; i < N; i++) P[i] = i;

    for (int k = 0; k < N; k++) {
        // Find pivot with cache-friendly search
        int max_row = k;
        double max_val = fabs(A[k][k]);
        #pragma omp simd reduction(max:max_val)
        for (int i = k + 1; i < N; i++) {
            double current = fabs(A[i][k]);
            if (current > max_val) {
                max_val = current;
                max_row = i;
            }
        }

        // Swap rows using cache-line aligned swaps
        if (max_row != k) {
            double temp[N] __attribute__((aligned(CACHE_LINE_SIZE)));
            memcpy(temp, A[k], sizeof(temp));
            memcpy(A[k], A[max_row], sizeof(temp));
            memcpy(A[max_row], temp, sizeof(temp));
            P[k] = max_row;
        }

        // Vectorized elimination with aligned accesses
        const double diag = A[k][k];
        #pragma omp simd aligned(A : CACHE_LINE_SIZE)
        for (int i = k + 1; i < N; i++) {
            A[i][k] /= diag;
        }

        // Tiled update to prevent false sharing
        const int tile_size = CACHE_LINE_SIZE / sizeof(double);
        for (int j = k + 1; j < N; j += tile_size) {
            int end = (j + tile_size < N) ? j + tile_size : N;
            #pragma omp simd aligned(A : CACHE_LINE_SIZE)
            for (int i = k + 1; i < N; i++) {
                const double factor = A[i][k];
                for (int jj = j; jj < end; jj++) {
                    A[i][jj] -= factor * A[k][jj];
                }
            }
        }
    }
}

int main() {
    // Set up thread topology - 4 batches with 12 threads each
    omp_set_nested(0);
    omp_set_max_active_levels(1);
    const int total_threads = 16;
    omp_set_num_threads(total_threads);

    // Read original matrix
    double** orig = read_matrix_from_csv_(
        "/home/pradyumn/Academic/Non_college/Main/Siemens/data/mytests/matrix_500x500.csv", N, N
    );

    // Create aligned batch with cache-friendly layout
    Matrix* batch = (Matrix*)aligned_alloc(CACHE_LINE_SIZE, NUM_MATRICES * sizeof(Matrix));
    
    // Initialize batch in parallel
    #pragma omp parallel for num_threads(total_threads) schedule(static)
    for (int m = 0; m < NUM_MATRICES; m++) {
        for (int i = 0; i < N; i++) {
            memcpy(batch[m].data[i], orig[i], N * sizeof(double));
        }
    }

    double start = omp_get_wtime();
    
    // Process matrices in parallel with thread-local decomposition
    #pragma omp parallel num_threads(total_threads)
    {
        #pragma omp single
        {
            int chunks = omp_get_num_threads();
            int chunk_size = NUM_MATRICES / chunks;
            
            for (int t = 0; t < chunks; t++) {
                const int start = t * chunk_size;
                const int end = (t == chunks-1) ? NUM_MATRICES : start + chunk_size;
                
                #pragma omp task firstprivate(start, end)
                {
                    for (int m = start; m < end; m++) {
                        lu_decomposition_single(&batch[m]);
                    }
                }
            }
        }
    }
    
    double elapsed = omp_get_wtime() - start;

    printf("Optimized execution time: %.4f seconds\n", elapsed);
    printf("Throughput: %.2f matrices/sec\n", NUM_MATRICES / elapsed);

    // Cleanup
    free(batch);
    for (int i = 0; i < N; i++) free(orig[i]);
    free(orig);

    return 0;
}