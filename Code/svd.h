#include "matrix.h"
#include <cusolverDn.h>
#include <cublas_v2.h> 
#include <cuda_runtime.h> 

void svd_parallel(double* d_H1, int k, double* d_S, double* d_U, double* d_VT);