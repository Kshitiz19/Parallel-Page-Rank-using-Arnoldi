#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>


__global__ void check(double* d_norm, int n, double* val)
{
    double temp = 0;
    double f ;
    for(int i=0; i<n; i++)
    {
        f = d_norm[i];
        temp += f*f;
    }
    *val = sqrt(temp);
    //if(*val <= 0) *val *= -1;
}

__global__ void calc_norm(double* d_A, double* d_q, int n, double* d_norm)
{
    int id = (blockIdx.x*blockDim.x) + threadIdx.x;
    if(id<n)
    {
        double temp = 0;
        int c = id*n;
        for(int i=0; i<n; i++, c++) temp += d_A[c]*d_q[i];
        d_norm[id] = temp - d_q[id];
    }
}

__global__ void saxpy(double *Q, double* v ,double* q, int n, int k)
{
    extern __shared__ double sv[];
    if(threadIdx.x == 0)
    {
        for(int i=0; i<k; i++) sv[i] = v[i];
    }
    __syncthreads();
    
    int id = (blockIdx.x*blockDim.x) + threadIdx.x;
    if(id < n){
        double temp = 0;
        int i = id*(k+1), j = i+k , count=-1;

        for(; i<j; i++) temp += Q[i]*sv[++count];
        q[id] = temp;
        }
}


void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++)
        {
            for(int col = 0 ; col < n ; col++)
                {
                    double Areg = A[row + col*lda];
                    printf(" %f ", Areg);
                }
            printf("\n");
        }
}

__global__ void get_v(double* d_VT, double* d_v, int k)
{
    int id = threadIdx.x;
    d_v[id] = d_VT[id*k +k-1+id];
}


__global__ void initCudaMat(double* A, int row, int col)
{
    for(int i=0; i<row*col; i++) A[i] = 0;
}

__global__ void printMatt(double* A, int row, int col)
{
    int z;
    for(z=0; z< (row*col); z++)
    {
        printf("%f  ", A[z]);
        if((z+1)%col == 0) printf("\n");
    }
}

__global__ void dev(int* col)
{
        printf("devinfo is %d\n", *col);
}
