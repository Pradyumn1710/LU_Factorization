#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double vector_infinity_norm(double* v, int n) {
    double max_val = 0.0;
    for (int i = 0; i < n; i++) {
        if (fabs(v[i]) > max_val) {
            max_val = fabs(v[i]);
        }
    }
    return max_val;
}


double compute_E1(double** A, double** L, double** U, int* P, int n) {
    // Step 1: Multiply L and U to get LU
    double** LU = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        LU[i] = (double*)calloc(n, sizeof(double));
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                LU[i][j] += L[i][k] * U[k][j];
            }
        }
    }

    // Step 2: Create PA (permuted A using P array)
    double** PA = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        PA[i] = (double*)malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            PA[i][j] = A[P[i]][j];  // apply permutation
        }
    }

    // Step 3: Compute infinity norm of (PA - LU)
    double max_error = 0.0;
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++) {
            row_sum += fabs(PA[i][j] - LU[i][j]);
        }
        if (row_sum > max_error) {
            max_error = row_sum;
        }
    }

    // Step 4: Free allocated memory
    for (int i = 0; i < n; i++) {
        free(LU[i]);
        free(PA[i]);
    }
    free(LU);
    free(PA);

    return max_error;
}


// Compute Error Metric E2 (Solution Accuracy)
double compute_E2(double* x, double* x_true, int n) {
    double* diff = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        diff[i] = x[i] - x_true[i];
    }
    double norm = vector_infinity_norm(diff, n);
    free(diff);
    return norm;
}


// Compute Error Metric E3 (Residual Norm)
double compute_E3(double** A, double* x, double* b, int n) {
    double* residual = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        residual[i] = b[i];
        for (int j = 0; j < n; j++) {
            residual[i] -= A[i][j] * x[j];
        }
    }
    double E3 = vector_infinity_norm(residual, n);
    free(residual);
    return E3;
}

