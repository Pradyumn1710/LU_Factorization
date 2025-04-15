#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>

#define MATRIX_SIZE 53        // Matrix dimension
#define NUM_THREADS 16        // CPU threads

// Function declarations
double** allocate_matrix();
void free_matrix(double **A);
void copy_matrix(double **dest, double **src);
void lu_decomposition_omp(double **A, int *P);
void solve_system_omp(double **A, int *P, double *b, double *x);
void apply_permutation(double *b, int *P, int n);
double** read_matrix_from_csv(const char* filename, int rows, int cols);
double* read_vector_from_csv(const char* filename, int n);
double vector_max_norm(double *v, int n);

// Matrix allocation utility
double** allocate_matrix() {
    double **A = (double **)malloc(MATRIX_SIZE * sizeof(double *));
    for(int i=0; i<MATRIX_SIZE; i++) {
        A[i] = (double *)malloc(MATRIX_SIZE * sizeof(double));
    }
    return A;
}

void free_matrix(double **A) {
    for(int i=0; i<MATRIX_SIZE; i++) free(A[i]);
    free(A);
}

void copy_matrix(double **dest, double **src) {
    for(int i=0; i<MATRIX_SIZE; i++) {
        memcpy(dest[i], src[i], MATRIX_SIZE * sizeof(double));
    }
}

// LU decomposition with partial pivoting
void lu_decomposition_omp(double **A, int *P) {
    for(int i=0; i<MATRIX_SIZE; i++) P[i] = i;

    for(int k=0; k<MATRIX_SIZE; k++) {
        // Find pivot
        int max_row = k;
        double max_val = fabs(A[k][k]);
        for(int i=k+1; i<MATRIX_SIZE; i++) {
            if(fabs(A[i][k]) > max_val) {
                max_val = fabs(A[i][k]);
                max_row = i;
            }
        }

        // Swap rows
        if(max_row != k) {
            double *tmp = A[k];
            A[k] = A[max_row];
            A[max_row] = tmp;
            int temp = P[k];
            P[k] = P[max_row];
            P[max_row] = temp;
        }

        // Elimination
        for(int i=k+1; i<MATRIX_SIZE; i++) {
            A[i][k] /= A[k][k];
            for(int j=k+1; j<MATRIX_SIZE; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
    }
}

// Apply permutation to vector
void apply_permutation(double *b, int *P, int n) {
    double *temp = (double *)malloc(n * sizeof(double));
    memcpy(temp, b, n * sizeof(double));
    for(int i=0; i<n; i++) b[i] = temp[P[i]];
    free(temp);
}

// Solve system
void solve_system_omp(double **A, int *P, double *b, double *x) {
    double y[MATRIX_SIZE];
    apply_permutation(b, P, MATRIX_SIZE);

    // Forward substitution
    for(int i=0; i<MATRIX_SIZE; i++) {
        y[i] = b[i];
        for(int j=0; j<i; j++) y[i] -= A[i][j] * y[j];
        y[i] /= A[i][i];
    }

    // Backward substitution
    for(int i=MATRIX_SIZE-1; i>=0; i--) {
        x[i] = y[i];
        for(int j=i+1; j<MATRIX_SIZE; j++) x[i] -= A[i][j] * x[j];
        x[i] /= A[i][i];
    }
}

// Error metric calculations
double compute_E1(double **A_orig, double **LU, int *P) {
    double max_err = 0.0;
    for(int i=0; i<MATRIX_SIZE; i++) {
        for(int j=0; j<MATRIX_SIZE; j++) {
            double reconstructed = 0.0;
            for(int k=0; k<=MIN(i,j); k++) {
                double l = (i > k) ? LU[i][k] : (i == k) ? 1.0 : 0.0;
                double u = (k <= j) ? LU[k][j] : 0.0;
                reconstructed += l * u;
            }
            double original = A_orig[P[i]][j];
            max_err = fmax(max_err, fabs(original - reconstructed));
        }
    }
    return max_err;
}

int main() {
    // Load data

    int n = MATRIX_SIZE;
    double** A = read_matrix_from_csv("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/A_matrix_case1.csv", n, n);
    double** A_copy = read_matrix_from_csv("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/A_matrix_case1.csv", n, n);
    // copy_matrix(A_copy, A);
    int *P = (int *)malloc(MATRIX_SIZE * sizeof(int));
    
    // LU decomposition
    lu_decomposition_omp(A, P);
    
    // E1: Factorization accuracy
    // double E1 = compute_E1(A_copy, A, P);
    
    // // E2: Solution accuracy
    // double *b =  read_matrix_from_csv("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/Case_1_soln.csv", n, 1);
    // double *x = read_matrix_from_csv("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/Case_1_soln.csv", n, 1);

    // solve_system_omp(A, P, b, x);
    
    // double E2 = 0.0;
    // for(int i=0; i<MATRIX_SIZE; i++) 
    //     E2 = fmax(E2, fabs(x[i] - 1.0));
    // E2 /= MATRIX_SIZE;  // Normalize by matrix size
    
    // // E3: Residual error
    // double *residual = (double *)malloc(MATRIX_SIZE * sizeof(double));
    // for(int i=0; i<MATRIX_SIZE; i++) {
    //     residual[i] = b[i];
    //     for(int j=0; j<MATRIX_SIZE; j++) 
    //         residual[i] -= A_copy[i][j] * x[j];
    // }
    // double E3 = vector_max_norm(residual, MATRIX_SIZE);
    
    // printf("========== Validation Results ==========\n");
    // printf("PA-LU Error (E1): %.3e\n", E1);
    // printf("Solution Error (E2): %.3e\n", E2);
    // printf("Residual Error (E3): %.3e\n", E3);
    
    // // Cleanup
    // free_matrix(A);
    // free_matrix(A_copy);
    // free(P);
    // free(b);
    // free(x);
    // free(residual);
    
    return 0;
}