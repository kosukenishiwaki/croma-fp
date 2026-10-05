#include <stdio.h>
#include <stdlib.h>

#include "tracer_fp_alloc.h"

double **allocate2DArray(int rows, int cols)
{
    double **array = (double **)malloc((size_t)rows * sizeof(double *));
    if (array == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for row pointers.\n");
        return NULL;
    }

    for (int i = 0; i < rows; i++) {
        array[i] = (double *)calloc((size_t)cols, sizeof(double));
        if (array[i] == NULL) {
            fprintf(stderr, "Error: Unable to allocate memory for row %d.\n", i);
            for (int j = 0; j < i; j++) {
                free(array[j]);
            }
            free(array);
            return NULL;
        }
    }

    return array;
}

double ***allocate3DArray(int d1, int d2, int d3)
{
    double ***array = (double ***)malloc((size_t)d1 * sizeof(double **));
    if (array == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for plane pointers.\n");
        return NULL;
    }

    for (int i = 0; i < d1; i++) {
        array[i] = (double **)malloc((size_t)d2 * sizeof(double *));
        if (array[i] == NULL) {
            fprintf(stderr, "Error: Unable to allocate memory for plane %d.\n", i);
            for (int j = 0; j < i; j++) {
                free(array[j]);
            }
            free(array);
            return NULL;
        }

        for (int j = 0; j < d2; j++) {
            array[i][j] = (double *)calloc((size_t)d3, sizeof(double));
            if (array[i][j] == NULL) {
                fprintf(stderr, "Error: Unable to allocate memory for row %d/%d.\n", i, j);
                for (int k = 0; k < j; k++) {
                    free(array[i][k]);
                }
                free(array[i]);
                for (int k = 0; k < i; k++) {
                    for (int l = 0; l < d2; l++) {
                        free(array[k][l]);
                    }
                    free(array[k]);
                }
                free(array);
                return NULL;
            }
        }
    }

    return array;
}

void free2Darray(double **array, int d1)
{
    if (array == NULL) return;

    for (int i = 0; i < d1; i++) {
        free(array[i]);
    }
    free(array);
}

void free3Darray(double ***array, int d1, int d2)
{
    if (array == NULL) return;

    for (int i = 0; i < d1; i++) {
        if (array[i] == NULL) continue;
        for (int j = 0; j < d2; j++) {
            free(array[i][j]);
        }
        free(array[i]);
    }
    free(array);
}
