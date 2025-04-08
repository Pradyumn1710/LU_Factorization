#include <stdio.h>
#include <stdlib.h>

int main() {
    // File paths
    const char *fileA = "Data/main/Case_A/A_matrix_case1.csv";
    const char *fileB = "Data/main/Case_A/B_matrix_case1.csv";
    const char *fileSoln = "Data/main/Case_A/Case_1_soln.csv";
    const char *fileError = "Data/main/Case_A/Error_file_case1.csv";

    // Open the first file
    FILE *fpA = fopen(fileA, "r");
    if (fpA == NULL) {
        perror("Error: Unable to open file A_matrix_case1.csv");
    } else {
        printf("Successfully opened file: %s\n", fileA);
        fclose(fpA);
    }

    // Open the second file
    FILE *fpB = fopen(fileB, "r");
    if (fpB == NULL) {
        perror("Error: Unable to open file B_matrix_case1.csv");
    } else {
        printf("Successfully opened file: %s\n", fileB);
        fclose(fpB);
    }

    // Open the solution file
    FILE *fpSoln = fopen(fileSoln, "r");
    if (fpSoln == NULL) {
        perror("Error: Unable to open file Case_1_soln.csv");
    } else {
        printf("Successfully opened file: %s\n", fileSoln);
        fclose(fpSoln);
    }

    // Open the error file
    FILE *fpError = fopen(fileError, "r");
    if (fpError == NULL) {
        perror("Error: Unable to open file Error_file_case1.csv");
    } else {
        printf("Successfully opened file: %s\n", fileError);
        fclose(fpError);
    }

    return 0;
}