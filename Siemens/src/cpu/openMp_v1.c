#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_MATRICES 10000    // Total number of LU factorizations
#define MATRIX_SIZE 53      // Matrix dimension
#define NUM_THREADS 16       // Number of threads (adjust as needed)

double **allocate_matrix();
void free_matrix(double **A);
void copy_matrix(double **dest, double **src);
void lu_decomposition(double **A);
void process_matrix(double **A, double *b, double *x);
void read_matrix_from_file(const char *filename, double **A, double *b, int n);
void forwardSubstitution(int n, double **L, double *b, double *y);
void backwardSubstitution(int n, double **U, double *y, double *x);
extern double** read_matrix_from_csv(const char* filename, int rows, int cols);
extern double vector_infinity_norm(double* v, int n);

extern double compute_E1(double** A, double** L, double** U, int* P, int n);

extern double compute_E2(double* x, double* x_true, int n);

extern double compute_E3(double** A, double* x, double* b, int n);

int main() {
    int n = MATRIX_SIZE;

    // /home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/A_matrix_case1.csv

    double** A = read_matrix_from_csv("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/A_matrix_case1.csv", n, n);
    double** A_copy = read_matrix_from_csv("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/A_matrix_case1.csv", n, n);
    double** B_matrix = read_matrix_from_csv("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/B_matrix_case1", n, 1);
    double** X_true_matrix = read_matrix_from_csv("/home/pradyumn/Academic/Non_college/Main/Siemens/data/main/Case_A/Case_1_soln.csv", n, 1);

    double* b = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) b[i] = B_matrix[i][0];

    double* x_true = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) x_true[i] = X_true_matrix[i][0];

    omp_set_num_threads(NUM_THREADS);
    double thread_times[NUM_THREADS] = {0};
    double total_start = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double thread_start = omp_get_wtime();

        double *x = (double*)malloc(n * sizeof(double));
        double *b_local = (double*)malloc(n * sizeof(double));
        double **A_local = allocate_matrix();

        #pragma omp for schedule(static)
        for (int i = 0; i < NUM_MATRICES; i++) {
            copy_matrix(A_local, A);
            for (int j = 0; j < n; j++) b_local[j] = b[j];
            process_matrix(A_local, b_local, x);
        }

        thread_times[tid] = omp_get_wtime() - thread_start;
        free(x);
        free(b_local);
        free_matrix(A_local);
    }

    double total_time = omp_get_wtime() - total_start;
    printf("\nTotal execution time: %.4f seconds\n", total_time);
    for (int t = 0; t < NUM_THREADS; t++) {
        printf("Thread %d time: %.4f seconds\n", t, thread_times[t]);
    }


    free_matrix(A);
    free_matrix(A_copy);
    free_matrix(B_matrix);
    free_matrix(X_true_matrix);
    free(b);
    free(x_true);

    return 0;
}


void process_matrix(double **A, double *b, double *x) {
    double **copy = allocate_matrix();
    copy_matrix(copy, A);
    lu_decomposition(copy);
    
    double y[MATRIX_SIZE];
    forwardSubstitution(MATRIX_SIZE, copy, b, y);
    backwardSubstitution(MATRIX_SIZE, copy, y, x);
    
    free_matrix(copy);
}

void lu_decomposition(double **A) {
    for (int k = 0; k < MATRIX_SIZE; k++) {
        for (int i = k+1; i < MATRIX_SIZE; i++) {
            A[i][k] /= A[k][k];
            for (int j = k+1; j < MATRIX_SIZE; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
    }
}

void forwardSubstitution(int n, double **L, double *b, double *y) {
    // #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++) {
            y[i] -= L[i][j] * y[j];
        }
        y[i] /= L[i][i];
    }
}

void backwardSubstitution(int n, double **U, double *y, double *x) {
    // #pragma omp parallel for
    for (int i = n - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= U[i][j] * x[j];
        }
        x[i] /= U[i][i];
    }
}

double **allocate_matrix() {
    double **A = (double **)malloc(MATRIX_SIZE * sizeof(double *));
    for (int i = 0; i < MATRIX_SIZE; i++) {
        A[i] = (double *)malloc(MATRIX_SIZE * sizeof(double));
    }
    return A;
}

void free_matrix(double **A) {
    for (int i = 0; i < MATRIX_SIZE; i++) free(A[i]);
    free(A);
}

void copy_matrix(double **dest, double **src) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            dest[i][j] = src[i][j];
        }
    }
}

void read_matrix_from_file(const char *filename, double **A, double *b, int n) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    char buffer[256];
    fgets(buffer, sizeof(buffer), file);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fscanf(file, "%lf", &A[i][j]);
        }
    }
    for (int i = 0; i < n; i++) {
        fscanf(file, "%lf", &b[i]);
    }
    fclose(file);
}