#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// -------------------- CSV Reading Utilities --------------------

// Reads a CSV file into a temporary 2D matrix (double**). 
// Returns NULL on error. The number of rows is stored in *rows and columns in *cols.
double** read_matrix_from_csv(const char* filename, int* rows, int* cols) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }
    char line[100000];
    int num_rows = 0;
    int num_cols = 0;
    int is_valid = 1;
    double** temp_matrix = NULL;
    int temp_capacity = 0;
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",\n");
        int current_cols = 0;
        double* row = NULL;
        while (token) {
            row = (double*)realloc(row, (current_cols + 1) * sizeof(double));
            row[current_cols++] = atof(token);
            token = strtok(NULL, ",\n");
        }
        if (current_cols == 0) continue;
        if (num_cols == 0) {
            num_cols = current_cols;
        } else if (current_cols != num_cols) {
            is_valid = 0;
            free(row);
            break;
        }
        if (num_rows >= temp_capacity) {
            temp_capacity = (temp_capacity == 0) ? 1 : temp_capacity * 2;
            temp_matrix = (double**)realloc(temp_matrix, temp_capacity * sizeof(double*));
        }
        temp_matrix[num_rows++] = row;
    }
    if (!is_valid || num_cols == 0) {
        for (int i = 0; i < num_rows; i++) free(temp_matrix[i]);
        free(temp_matrix);
        fclose(file);
        return NULL;
    }
    *rows = num_rows;
    *cols = num_cols;
    fclose(file);
    return temp_matrix;
}

// Reads a CSV file into a vector. Returns NULL on error.
double* read_vector_from_csv(const char* filename, int* size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }
    char line[10000];
    double* vector = NULL;
    int capacity = 0;
    *size = 0;
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",\n");
        while (token) {
            if (*size >= capacity) {
                capacity = (capacity == 0) ? 1 : capacity * 2;
                vector = (double *)realloc(vector, capacity * sizeof(double));
            }
            vector[(*size)++] = atof(token);
            token = strtok(NULL, ",\n");
        }
    }
    fclose(file);
    return vector;
}

// Frees a 2D matrix allocated by read_matrix_from_csv.
void free_matrix_2d(double** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// -------------------- CUDA Kernels --------------------

// Kernel to compute the multipliers for column k of L for rows i = k+1...n-1.
__global__ void computeLKernel(double* U, double* L, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    if(i < n) {
        L[i * n + k] = U[i * n + k] / U[k * n + k];
    }
}

// Kernel to update U for rows i = k+1...n-1 and columns j = k...n-1.
__global__ void updateUKernel(double* U, double* L, int k, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + k + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + k;
    if(i < n && j < n) {
        U[i * n + j] -= L[i * n + k] * U[k * n + j];
    }
}

// -------------------- LU Decomposition (CUDA) --------------------

// Performs LU decomposition with partial pivoting on a flattened matrix A.
// The matrices L and U are stored in unified memory and use row-major order.
// The permutation vector P is maintained on the CPU.
void lu_decomposition_cuda(double* A, double* L, double* U, int* P, int n) {
    // Initialize L as identity and U as a copy of A.
    for (int i = 0; i < n; i++) {
        P[i] = i;
        for (int j = 0; j < n; j++) {
            if (i == j)
                L[i * n + j] = 1.0;
            else
                L[i * n + j] = 0.0;
            U[i * n + j] = A[i * n + j];
        }
    }

    // Loop over each column for elimination.
    for (int k = 0; k < n; k++) {
        // Partial pivoting on CPU.
        double max_val = fabs(U[k * n + k]);
        int max_idx = k;
        for (int i = k + 1; i < n; i++) {
            double abs_val = fabs(U[i * n + k]);
            if (abs_val > max_val) {
                max_val = abs_val;
                max_idx = i;
            }
        }
        if (max_idx != k) {
            // Swap rows k and max_idx in U.
            for (int j = 0; j < n; j++) {
                double temp = U[k * n + j];
                U[k * n + j] = U[max_idx * n + j];
                U[max_idx * n + j] = temp;
            }
            // Swap corresponding rows in L for columns 0 .. k-1.
            for (int j = 0; j < k; j++) {
                double temp = L[k * n + j];
                L[k * n + j] = L[max_idx * n + j];
                L[max_idx * n + j] = temp;
            }
            // Update permutation vector.
            int temp = P[k];
            P[k] = P[max_idx];
            P[max_idx] = temp;
        }

        // Launch kernel to compute L values for current column k.
        int blockSize = 256;
        int gridSize = (n - k - 1 + blockSize - 1) / blockSize;
        computeLKernel<<<gridSize, blockSize>>>(U, L, k, n);
        cudaDeviceSynchronize();

        // Launch kernel to update U.
        dim3 blockDim(16, 16);
        dim3 gridDim((n - k + blockDim.x - 1) / blockDim.x, (n - k - 1 + blockDim.y - 1) / blockDim.y);
        updateUKernel<<<gridDim, blockDim>>>(U, L, k, n);
        cudaDeviceSynchronize();
    }
}

// -------------------- Utility Functions --------------------

// Multiply matrix A (flattened, n x n) by vector x, result in y.
void matrix_vector_multiply(double* A, double* x, double* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = 0.0;
        for (int j = 0; j < n; j++) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

// Infinity norm for a vector.
double infinity_norm_vector(double* vector, int size) {
    double max = 0.0;
    for (int i = 0; i < size; i++) {
        double abs_val = fabs(vector[i]);
        if (abs_val > max) {
            max = abs_val;
        }
    }
    return max;
}

// Infinity norm for a matrix (flattened, n x n): maximum row sum.
double infinity_norm_matrix(double* matrix, int n) {
    double max_row_sum = 0.0;
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++) {
            row_sum += fabs(matrix[i * n + j]);
        }
        if (row_sum > max_row_sum) {
            max_row_sum = row_sum;
        }
    }
    return max_row_sum;
}

// Compute error metric E1 = || PA - LU ||_inf.
double calculate_E1(double* A, double* L, double* U, int* P, int n) {
    double* PA = (double*)malloc(n * n * sizeof(double));
    double* LU = (double*)malloc(n * n * sizeof(double));
    double* diff = (double*)malloc(n * n * sizeof(double));

    // Compute PA (permute A using P).
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            PA[i * n + j] = A[P[i] * n + j];
        }
    }
    // Compute LU = L * U.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            LU[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                double l_val = (i == k) ? 1.0 : (i > k ? L[i * n + k] : 0.0);
                double u_val = (k <= j) ? U[k * n + j] : 0.0;
                LU[i * n + j] += l_val * u_val;
            }
        }
    }
    // Compute difference PA - LU.
    for (int i = 0; i < n * n; i++) {
        diff[i] = PA[i] - LU[i];
    }
    double norm = infinity_norm_matrix(diff, n);
    free(PA);
    free(LU);
    free(diff);
    return norm;
}

// Compute error metric E2 = || b - A*x ||_inf.
double calculate_E2(double* A, double* x, double* b, int n) {
    double* Ax = (double*)malloc(n * sizeof(double));
    double* diff = (double*)malloc(n * sizeof(double));
    matrix_vector_multiply(A, x, Ax, n);
    for (int i = 0; i < n; i++) {
        diff[i] = b[i] - Ax[i];
    }
    double norm = infinity_norm_vector(diff, n);
    free(Ax);
    free(diff);
    return norm;
}

// Compute error metric E3 = || x - x_true ||_inf.
double calculate_E3(double* x, double* x_true, int n) {
    double* diff = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        diff[i] = x[i] - x_true[i];
    }
    double norm = infinity_norm_vector(diff, n);
    free(diff);
    return norm;
}

// Solve the system LUx = Pb using forward and backward substitution.
// L and U are flattened matrices (n x n) stored in unified memory.
void solve_lu_system(double* L, double* U, int* P, double* b, double* x, int n) {
    double* y = (double*)malloc(n * sizeof(double));
    // Forward substitution: solve L*y = Pb.
    for (int i = 0; i < n; i++) {
        y[i] = b[P[i]];
        for (int j = 0; j < i; j++) {
            y[i] -= L[i * n + j] * y[j];
        }
    }
    // Backward substitution: solve U*x = y.
    for (int i = n - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= U[i * n + j] * x[j];
        }
        x[i] /= U[i * n + i];
    }
    free(y);
}

// -------------------- Test Cases --------------------

// Test Case A: Reads matrix A, vector b and true solution x_true from CSV files,
// flattens A, performs LU decomposition on the GPU, and computes error metrics.
void test_case_A() {
    int n, n_check;
    // Read A as a 2D matrix.
    double** A_2d = read_matrix_from_csv("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/A_matrix_case1.csv", &n, &n_check);
    if (!A_2d || n <= 0) {
        printf("Error reading A matrix!\n");
        return;
    }
    if (n != n_check) {
        printf("Error: A matrix is not square!\n");
        free_matrix_2d(A_2d, n);
        return;
    }
    // Flatten A into a 1D array.
    double* A = (double*)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = A_2d[i][j];
        }
    }
    free_matrix_2d(A_2d, n);
    
    // Read vector b.
    double* b = read_vector_from_csv("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/B_matrix_case1.csv", &n_check);
    if (!b || n_check != n) {
        printf("Error reading B vector!\n");
        free(A);
        return;
    }
    // Read true solution.
    double* x_true = read_vector_from_csv("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/Case_1_soln.csv", &n_check);
    if (!x_true || n_check != n) {
        printf("Error reading true solution!\n");
        free(A);
        free(b);
        return;
    }
    
    // Allocate unified memory for L and U.
    double *L, *U;
    cudaMallocManaged(&L, n * n * sizeof(double));
    cudaMallocManaged(&U, n * n * sizeof(double));
    
    // Allocate and initialize permutation vector.
    int* P = (int*)malloc(n * sizeof(int));
    
    // Perform LU decomposition using CUDA.
    lu_decomposition_cuda(A, L, U, P, n);
    
    double E1 = calculate_E1(A, L, U, P, n);
    printf("Test Case A:\nError Metric 1 (E1): %.2e\n", E1);
    
    // Create vector of ones.
    double* ones = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) ones[i] = 1.0;
    
    // Compute row sums of A.
    double* row_sums = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        row_sums[i] = 0.0;
        for (int j = 0; j < n; j++) {
            row_sums[i] += A[i * n + j];
        }
    }
    
    double* x_computed = (double*)malloc(n * sizeof(double));
    solve_lu_system(L, U, P, row_sums, x_computed, n);
    
    double E2 = calculate_E3(x_computed, ones, n);
    printf("Error Metric 2 (E2): %.2e\n", E2);
    
    double* x_solved = (double*)malloc(n * sizeof(double));
    solve_lu_system(L, U, P, b, x_solved, n);
    
    double E3 = calculate_E3(x_solved, x_true, n);
    printf("Error Metric 3 (E3): %.2e\n", E3);
    
    // Free all memory.
    free(A);
    free(b);
    free(x_true);
    free(ones);
    free(row_sums);
    free(x_computed);
    free(x_solved);
    free(P);
    cudaFree(L);
    cudaFree(U);
}

// Test Case B: Similar to Test Case A but handles two different B vectors and corresponding solutions.
void test_case_B() {
    printf("\n----- Running Test Case B -----\n\n");
    int n, n_check;
    
    double** A_2d = read_matrix_from_csv("Case B/Case B - files/A_matrix.csv", &n, &n_check);
    if (!A_2d || n <= 0 || n != n_check) {
        printf("Error reading A matrix for Case B!\n");
        return;
    }
    printf("Successfully read A matrix of size %d x %d\n", n, n);
    
    // Flatten A.
    double* A = (double*)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i * n + j] = A_2d[i][j];
    free_matrix_2d(A_2d, n);
    
    // Allocate unified memory for L and U.
    double *L, *U;
    cudaMallocManaged(&L, n * n * sizeof(double));
    cudaMallocManaged(&U, n * n * sizeof(double));
    
    int* P = (int*)malloc(n * sizeof(int));
    
    // Perform LU decomposition.
    lu_decomposition_cuda(A, L, U, P, n);
    
    double E1 = calculate_E1(A, L, U, P, n);
    printf("Error Metric 1 (E1): %.2e\n", E1);
    
    // ----- Solve for first B vector -----
    int b1_size;
    double* b1 = read_vector_from_csv("Case B/Case B - files/B1_case2.csv", &b1_size);
    if (!b1 || b1_size != n) {
        printf("Error reading first B vector!\n");
        free(A);
        free(P);
        cudaFree(L);
        cudaFree(U);
        return;
    }
    printf("Successfully read first B vector of size %d\n", b1_size);
    
    int x1_true_size;
    double* x1_true = read_vector_from_csv("Case B/Case B - files/U_solution.csv", &x1_true_size);
    if (!x1_true || x1_true_size != n) {
        printf("Error reading first true solution vector!\n");
        free(b1);
        free(A);
        free(P);
        cudaFree(L);
        cudaFree(U);
        return;
    }
    printf("Successfully read first true solution vector of size %d\n", x1_true_size);
    
    double* x1_solved = (double*)malloc(n * sizeof(double));
    solve_lu_system(L, U, P, b1, x1_solved, n);
    
    double E2_B1 = calculate_E2(A, x1_solved, b1, n);
    double E3_B1 = calculate_E3(x1_solved, x1_true, n);
    
    printf("\nResults for first B vector:\n");
    printf("Error Metric 2 (E2): %.2e\n", E2_B1);
    printf("Error Metric 3 (E3): %.2e\n", E3_B1);
    
    // ----- Solve for second B vector -----
    int b2_size;
    double* b2 = read_vector_from_csv("Case B/Case B - files/B2_case2.csv", &b2_size);
    if (!b2 || b2_size != n) {
        printf("Error reading second B vector!\n");
        free(b1);
        free(x1_true);
        free(x1_solved);
        free(A);
        free(P);
        cudaFree(L);
        cudaFree(U);
        return;
    }
    printf("\nSuccessfully read second B vector of size %d\n", b2_size);
    
    int x2_true_size;
    double* x2_true = read_vector_from_csv("Case B/Case B - files/U_solution_2.csv", &x2_true_size);
    if (!x2_true || x2_true_size != n) {
        printf("Error reading second true solution vector!\n");
        free(b1);
        free(x1_true);
        free(x1_solved);
        free(b2);
        free(A);
        free(P);
        cudaFree(L);
        cudaFree(U);
        return;
    }
    printf("Successfully read second true solution vector of size %d\n", x2_true_size);
    
    double* x2_solved = (double*)malloc(n * sizeof(double));
    solve_lu_system(L, U, P, b2, x2_solved, n);
    
    double E2_B2 = calculate_E2(A, x2_solved, b2, n);
    double E3_B2 = calculate_E3(x2_solved, x2_true, n);
    
    printf("\nResults for second B vector:\n");
    printf("Error Metric 2 (E2): %.2e\n", E2_B2);
    printf("Error Metric 3 (E3): %.2e\n", E3_B2);
    
    // Clean up.
    free(b1);
    free(x1_true);
    free(x1_solved);
    free(b2);
    free(x2_true);
    free(x2_solved);
    free(A);
    free(P);
    cudaFree(L);
    cudaFree(U);
    
    printf("\n----- Test Case B Completed -----\n");
}

// -------------------- Main Function --------------------

int main() {
    printf("Running LU Decomposition Test Case A using CUDA...\n");
    test_case_A();
    printf("Test Case A completed.\n\n");
    
    test_case_B();
    
    return 0;
}