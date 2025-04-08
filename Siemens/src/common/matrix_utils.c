#include<stdio.h>
#include<stdlib.h>

void swap_rows(double** A, int* P, int i, int j) {
    double* temp = A[i];
    A[i] = A[j];
    A[j] = temp;

    int tempP = P[i];
    P[i] = P[j];
    P[j] = tempP;
}
