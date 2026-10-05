#ifndef TRACER_FP_ALLOC_H
#define TRACER_FP_ALLOC_H

double **allocate2DArray(int rows, int cols);
double ***allocate3DArray(int d1, int d2, int d3);
void free2Darray(double **array, int d1);
void free3Darray(double ***array, int d1, int d2);

#endif
