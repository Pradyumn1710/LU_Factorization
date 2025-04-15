#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define NUM_MATRICES 100   // Number of LU factorizations to perform
#define MATRIX_SIZE 1000   // Size of each matrix
#define NUM_THREADS 16     // Adjust to match your CPU cores

// Allocate a contiguous 2D matrix
double** allocate_matrix() {
    double **A = malloc(MATRIX_SIZE * sizeof(double*));
    double *data = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    if (!A || !data) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < MATRIX_SIZE; i++) {
        A[i] = data + i * MATRIX_SIZE;
    }
    return A;
}

void free_matrix(double **A) {
    if (A) {
        free(A[0]);
        free(A);
    }
}

void copy_matrix(double **dest, double **src) {
    memcpy(dest[0], src[0], MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
}

// Core LU decomposition (no pivoting)
void lu_decomposition(double **A) {
    for (int k = 0; k < MATRIX_SIZE; k++) {
        double pivot = A[k][k];
        for (int i = k + 1; i < MATRIX_SIZE; i++) {
            A[i][k] /= pivot;
            for (int j = k + 1; j < MATRIX_SIZE; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
    }
}

// Efficient CSV reader (assumes no header and comma-separated)
double** read_matrix_from_csv_(const char* filename, int rows, int cols) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    double** matrix = allocate_matrix();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%lf", &matrix[i][j]) != 1) {
                fprintf(stderr, "Error reading matrix element at (%d, %d)\n", i, j);
                exit(EXIT_FAILURE);
            }
            if (j < cols - 1) fgetc(file);  // Skip comma
        }
    }

    fclose(file);
    return matrix;
}

int main() {
    omp_set_num_threads(NUM_THREADS);

    double** base_matrix = read_matrix_from_csv_("/home/pradyumn/Academic/Non_college/Main/Siemens/data/mytests/matrix_1000x1000.csv", MATRIX_SIZE, MATRIX_SIZE);
    double total_time = 0.0;

    double thread_times[NUM_THREADS] = {0};

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double local_time = 0.0;

        // Each thread gets its own matrix
        double** local_matrix = allocate_matrix();

        #pragma omp for schedule(static)
        for (int m = 0; m < NUM_MATRICES; m++) {
            copy_matrix(local_matrix, base_matrix);

            double start = omp_get_wtime();
            lu_decomposition(local_matrix);
            double end = omp_get_wtime();

            local_time += (end - start);
        }

        thread_times[tid] = local_time;

        #pragma omp atomic
        total_time += local_time;

        free_matrix(local_matrix);
    }

    printf("OpenMP Batched LU Decomposition (%d matrices, %d x %d):\n", NUM_MATRICES, MATRIX_SIZE, MATRIX_SIZE);
    printf("Total time: %.6f sec | Average per matrix: %.6f ms\n", total_time, (total_time / NUM_MATRICES) * 1000.0);

    for (int t = 0; t < NUM_THREADS; t++) {
        printf("Thread %2d time: %.6f sec\n", t, thread_times[t]);
    }

    free_matrix(base_matrix);
    return 0;
}
