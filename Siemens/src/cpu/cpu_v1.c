#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <time.h>


// Import the swap_rows function from matrix_utils.c
extern void swap_rows(double** matrix, int* P, int row1, int row2);

// Function to swap two rows and update the pivot array
extern double** read_matrix_from_csv(const char* filename, int rows, int cols);

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
            swap_rows(A, P, k, max_index);
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
void extract_LU(double** A, double** L, double** U, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i > j) {
                L[i][j] = A[i][j];
                U[i][j] = 0.0;
            } else if (i == j) {
                L[i][j] = 1.0;

                U[i][j] = A[i][j];
            } else {
                L[i][j] = 0.0;
                U[i][j] = A[i][j];
            }
        }
    }
}

// Forward substitution: Solves Ly = Pb
void forward_substitution(double** L, double* b, double* y, int* P, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = b[P[i]];
        for (int j = 0; j < i; j++) {
            y[i] -= L[i][j] * y[j];
        }
    }
}

// Backward substitution: Solves Ux = y
void backward_substitution(double** U, double* y, double* x, int n) {
    for (int i = n - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= U[i][j] * x[j];
        }
        x[i] /= U[i][i];
    }
}

// Compute infinity norm
extern double vector_infinity_norm(double* v, int n);

extern double compute_E1(double** A, double** L, double** U, int* P, int n);

extern double compute_E2(double* x, double* x_true, int n);

extern double compute_E3(double** A, double* x, double* b, int n);

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
            printf("%.20f ", matrix[i][j]); 
        }
        printf("\n"); 
    }
} 


int main() {
    int n = 53;

    double** A = read_matrix_from_csv("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/A_matrix_case1.csv", n, n);
    double** A_copy = read_matrix_from_csv("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/A_matrix_case1.csv", n, n);
    double** B_matrix = read_matrix_from_csv("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/B_matrix_case1.csv", n, 1);
    double** X_true_matrix = read_matrix_from_csv("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/Case_1_soln.csv", n, 1);

    // Extract vectors
    double* b = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        b[i] = B_matrix[i][0];
    }
    double* x_true = (double*)malloc(n * sizeof(double));
    // Uncomment and initialize x_true if needed
    for (int i = 0; i < n; i++) {
        x_true[i] = X_true_matrix[i][0];
    }
    
    clock_t start_time = clock();

    // LU decomposition
    int* P = (int*)malloc(n * sizeof(int));
    lu_factorization(A, P, n);

    clock_t end_time = clock();

    double time_taken = (((double)(end_time-start_time))/CLOCKS_PER_SEC);
    printf("Time taken by Lu is %.12f seconds",time_taken);
    // Extract L and U
    double** L = (double**)malloc(n * sizeof(double*));
    double** U = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        L[i] = (double*)malloc(n * sizeof(double));
        U[i] = (double*)malloc(n * sizeof(double));
    }
    extract_LU(A, L, U, n);
    
    // ------ Critical Fix: Solve for x ------
    double* y = (double*)malloc(n * sizeof(double));
    double* x = (double*)malloc(n * sizeof(double));
    

    forward_substitution(L, b, y, P, n);  // Uses permutation P internally
    backward_substitution(U, y, x, n);

    // Compute error metrics
    double E1 = compute_E1(A_copy, L, U, P, n);
    double E2 = compute_E2(x, x_true, n);  // Pass computed x and x_true
    double E3 = compute_E3(A_copy, x, b, n);  // Pass computed x and original b

    printf("\nE1 (Factorization Accuracy): %.12e\n", E1);
    printf("E2 (Solution Accuracy): %.12e\n", E2);
    printf("E3 (Residual Norm): %.12e\n", E3);


    // Free all allocated memory (add cleanup code here)

    return 0;
}
