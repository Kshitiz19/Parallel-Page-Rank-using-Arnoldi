__global__ void check(double* d_norm, int n, double* val);
__global__ void calc_norm(double* d_A, double* d_q, int n, double* d_norm);
__global__ void saxpy(double *Q, double* v ,double* q, int n, int k);
void printMatrix(int m, int n, const double*A, int lda, const char* name);
__global__ void get_v(double* d_VT, double* d_v, int k);
__global__ void initCudaMat(double* A, int row, int col);
__global__ void printMatt(double* A, int row, int col);
__global__ void dev(int* col);
