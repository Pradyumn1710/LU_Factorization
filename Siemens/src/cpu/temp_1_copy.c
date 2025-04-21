#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define NUM_MATRICES 100
#define MATRIX_SIZE 1000
#define OUTER_THREADS 4
#define INNER_THREADS 3
#define CACHE_LINE 64

typedef struct {
    double* data;      // Flattened matrix data (row-major)
    int* perm;         // Permutation vector
} Matrix;

// Allocate aligned memory for matrix
Matrix create_matrix(int size) {
    Matrix mat;
    posix_memalign((void**)&mat.data, CACHE_LINE, size * size * sizeof(double));
    posix_memalign((void**)&mat.perm, CACHE_LINE, size * sizeof(int));
    return mat;
}

// Read matrix from file into contiguous memory
Matrix read_matrix_from_csv_(const char* filename, int size) {
    Matrix mat = create_matrix(size);
    FILE* file = fopen(filename, "r");
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fscanf(file, "%lf,", &mat.data[i*size + j]);
        }
    }
    fclose(file);
    return mat;
}

// LU decomposition with nested parallelism
void lu_decomposition(Matrix* matrix, int size) {
    double* A = matrix->data;
    int* P = matrix->perm;
    
    // Initialize permutation vector
    for (int i = 0; i < size; i++) P[i] = i;

    for (int k = 0; k < size; k++) {
        // Find pivot with SIMD optimization
        int max_row = k;
        double max_val = fabs(A[k*size + k]);
        
        #pragma omp simd reduction(max:max_val)
        for (int i = k+1; i < size; i++) {
            double current = fabs(A[i*size + k]);
            if (current > max_val) {
                max_val = current;
                max_row = i;
            }
        }

        // Swap rows using cache-friendly blocks
        if (max_row != k) {
            #pragma omp simd aligned(A:CACHE_LINE)
            for (int j = 0; j < size; j++) {
                double tmp = A[k*size + j];
                A[k*size + j] = A[max_row*size + j];
                A[max_row*size + j] = tmp;
            }
            P[k] = max_row;
        }

        // Nested parallel elimination
        #pragma omp parallel num_threads(INNER_THREADS)
        {
            #pragma omp for schedule(static) nowait
            for (int i = k+1; i < size; i++) {
                const double factor = A[i*size + k] / A[k*size + k];
                A[i*size + k] = factor;

                #pragma omp simd aligned(A:CACHE_LINE)
                for (int j = k+1; j < size; j++) {
                    A[i*size + j] -= factor * A[k*size + j];
                }
            }
        }
    }
}

int main() {
    // Configure nested parallelism
    omp_set_nested(1);
    omp_set_max_active_levels(2);
    omp_set_num_threads(OUTER_THREADS);

    // Read master matrix
    Matrix master = read_matrix_from_csv_(
        "/home/pradyumn/Academic/Non_college/Main/Siemens/data/mytests/matrix_500x500.csv", MATRIX_SIZE
    );

    // Create matrix batch with contiguous memory
    Matrix* batch = (Matrix*)aligned_alloc(CACHE_LINE, NUM_MATRICES * sizeof(Matrix));
    #pragma omp parallel for num_threads(OUTER_THREADS)
    for (int m = 0; m < NUM_MATRICES; m++) {
        batch[m] = create_matrix(MATRIX_SIZE);
        memcpy(batch[m].data, master.data, 
              MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
        memcpy(batch[m].perm, master.perm, MATRIX_SIZE * sizeof(int));
    }

    // Benchmark
    double start = omp_get_wtime();
    #pragma omp parallel for num_threads(OUTER_THREADS) schedule(static)
    for (int m = 0; m < NUM_MATRICES; m++) {
        lu_decomposition(&batch[m], MATRIX_SIZE);
    }
    double time = omp_get_wtime() - start;

    printf("Optimized time: %.4f seconds\n", time);
    printf("Speedup: %.2fx\n", (NUM_MATRICES*0.02056)/time);

    // Cleanup
    free(master.data);
    free(master.perm);
    #pragma omp parallel for
    for (int m = 0; m < NUM_MATRICES; m++) {
        free(batch[m].data);
        free(batch[m].perm);
    }
    free(batch);

    return 0;
}