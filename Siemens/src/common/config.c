#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to read a matrix from a CSV file
double** read_matrix_from_csv(const char* filename, int rows, int cols) {
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

// Reads a matrix from CSV into a 1D array of size rows * cols
double* read_matrix_from_csv_flat(const char* filename, int rows, int cols) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file '%s'\n", filename);
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    double* matrix = (double*)malloc(rows * cols * sizeof(double));
    if (!matrix) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows * cols; i++) {
        if (fscanf(file, "%lf", &matrix[i]) != 1) {
            fprintf(stderr, "Error reading matrix element %d\n", i);
            exit(EXIT_FAILURE);
        }
        fgetc(file); // Consume comma or newline
    }

    fclose(file);
    return matrix;
}

