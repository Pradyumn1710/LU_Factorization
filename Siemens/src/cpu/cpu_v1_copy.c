#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <time.h>

// Include all files from the .common folder


// Import the swap_rows function from matrix_utils.c
void swap_rows_(double** A, int* P, int i, int j) {
    double* temp = A[i];
    A[i] = A[j];
    A[j] = temp;

    int tempP = P[i];
    P[i] = P[j];
    P[j] = tempP;
}

// Function to swap two rows and update the pivot array
double** read_matrix_from_csv_(const char* filename, int rows, int cols) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file '%s'\n", filename);
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    double** matrix = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%lf", &matrix[i][j]) != 1) {
                perror("Failed to read matrix element");
                fprintf(stderr, "Error reading row %d, col %d\n", i, j);
                exit(EXIT_FAILURE);
            }
            if (j < cols - 1) fgetc(file);  // Consume the comma
        }
    }

    fclose(file);
    return matrix;
}
// LU factorization with partial pivoting
void lu_factorization(double** A, int* P, int n) {
    // Initialised P matrix here , it accounts for swaps
    for (int i = 0; i < n; i++) {
        P[i] = i;
    }
    //Row swapping (pivoting) is done here 
    for (int k = 0; k < n - 1; k++) {
        int max_index = k;
        for (int i = k + 1; i < n; i++) {
            if (fabs(A[i][k]) > fabs(A[max_index][k])) {
                max_index = i;
            }
        }

        if (max_index != k) {
            swap_rows_(A, P, k, max_index);
        }

        for (int i = k + 1; i < n; i++) {
            if (fabs(A[k][k]) < 1e-12) {
                fprintf(stderr, "Error: Zero pivot detected at A[%d][%d]\n", k, k);
                exit(EXIT_FAILURE);
            }
            A[i][k] /= A[k][k];
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
    }
}

// Extract L and U from A after LU factorization

// Main function

void check_file_exists(const char *path) {
    struct stat buffer;
    if (stat(path, &buffer) == 0) {
        printf("File exists: %s\n", path);
    } else {
        perror("stat error");
        printf("File does not exist: %s\n", path);
        exit(1);
    }
}
void print_matrix(double** matrix, int rows, int cols) {
    printf("Matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.20f ", matrix[i][j]); // Print each element with 6 decimal precision
        }
        printf("\n"); // Move to the next row
    }
} 


int main() {
    int n = 1000;
    int N = 500; // Number of LU runs

    // double * h_A = read_matrix_from_csv_flat("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_B/A_matrix.csv", n, n);

    // Load master matrix once
    const char* A_path = "/home/pradyumn/Academic/Non_college/Main/Siemens/data/mytests/lu_decomposable_matrix_1000x1000.csv";
    double** A_master = read_matrix_from_csv_(A_path, n, n);

    // Preallocate N copies of A
    double*** A_batch = (double***)malloc(N * sizeof(double**));
    for (int k = 0; k < N; k++) {
        A_batch[k] = (double**)malloc(n * sizeof(double*));
        for (int i = 0; i < n; i++) {
            A_batch[k][i] = (double*)malloc(n * sizeof(double));
            for (int j = 0; j < n; j++) {
                A_batch[k][i][j] = A_master[i][j];  // Copy before timing
            }
        }
    }

    // Allocate pivot arrays
    int** P_batch = (int**)malloc(N * sizeof(int*));
    for (int k = 0; k < N; k++) {
        P_batch[k] = (int*)malloc(n * sizeof(int));
    }

    // ---------- Benchmark Starts Here ----------
    clock_t start_time = clock();

    for (int k = 0; k < N; k++) {
        lu_factorization(A_batch[k], P_batch[k], n);
    }

    clock_t end_time = clock();
    // ---------- Benchmark Ends Here ------------

    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("LU factorization on %d matrices:\n", N);
    printf("Total LU time: %.6f seconds\n", time_taken);
    printf("Average LU time: %.6f ms\n", (time_taken / N) * 1000.0);

    // Cleanup
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < n; i++) {
            free(A_batch[k][i]);
        }
        free(A_batch[k]);
        free(P_batch[k]);
    }
    free(A_batch);
    free(P_batch);

    for (int i = 0; i < n; i++) {
        free(A_master[i]);
    }
    free(A_master);

    return 0;
}
