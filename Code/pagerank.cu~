#include <stdio.h>
#include <stdlib.h>
#include "arnoldi.h"
#include "pageRankKernels.h"
#include "matrix.h"
#include <math.h>
#include <string.h>
#include "filereader.h"
#include "svd.h"
#include <cuda_runtime.h>

using namespace std;

int main(int argc, char** argv)
{
    const int NUM_THREADS = 256;
    
    cudaError_t err = cudaSuccess;
    int k = 8;			// experiment with value of k(power of 2 only) for optimality
    int n;
    double tempNorm;

    Matrix A;

    double *d_norm;
    double *d_q1;
    double *d_H1;
    double *d_Q;
    double *temp, *d_z;
    double* d_A, *h_A, *h_q;
    double *d_v;


    readInput(&A, argv[1], 0.85);
    n = A.A_rows;
    printf("n = %d\n", n);

    matrixTranspose(&A);

    double **AMat = A.A;
    h_A = (double*)malloc(sizeof(double)*n*n);
    h_q = (double*)malloc(sizeof(double)*n);

    double val = (1.0/sqrt(n));
    for(int i=0, c=0; i<n; i++)
    {
        for(int j=0; j<n; j++, c++) h_A[c] = AMat[i][j];
        h_q[i] = val;
    }

    cusolverDnHandle_t cusolverH = NULL;

    const int rows = k+1;
    const int cols = k;
    const int lda = rows;
/*
    double U[lda*rows];
    double VT[lda*cols];
    double S[cols];
*/
    double *d_S = NULL;
    double *d_U = NULL;
    double *d_VT = NULL;
    int *devInfo = NULL;
    double *d_work = NULL;
    double *d_rwork = NULL;
    int lwork = 0;

    cudaMalloc ((void**)&d_S , sizeof(double)*cols);
    cudaMalloc ((void**)&d_U , sizeof(double)*lda*rows);
    cudaMalloc ((void**)&d_VT , sizeof(double)*lda*cols);
    cudaMalloc ((void**)&devInfo, sizeof(int));

    cusolverDnCreate(&cusolverH);
    cusolverDnDgesvd_bufferSize( cusolverH, rows, cols, &lwork );
    cudaMalloc((void**)&d_work , sizeof(double)*lwork);

    cudaMalloc((void**) &d_norm, sizeof(double)*n);
    cudaMalloc((void**) &d_A, sizeof(double)*n*n);
    cudaMalloc((void**) &d_H1, sizeof(double)*k*(k+1));
    cudaMalloc((void**) &d_q1, sizeof(double)*n);
    cudaMalloc((void**) &d_Q, sizeof(double)*n*(k+1));
    cudaMalloc((void**) &temp, sizeof(double));
    cudaMalloc((void**) &d_z, sizeof(double)*n);
    cudaMalloc((void**) &d_v, sizeof(double)*k);
    //cudaMalloc((void**) &d_q2, sizeof(double)*n);


    cudaMemcpy(d_A, h_A, n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q1, h_q, n*sizeof(double), cudaMemcpyHostToDevice);

    initCudaMat<<<1, 1>>>(d_H1, k+1, k);
    initCudaMat<<<1, 1>>>(d_Q, n, k+1);
    initCudaMat<<<1, 1>>>(d_z, n, 1);

    for(int fold = 0; fold < 10; fold++)
    {
        //printf("Fold = %d\n\n", fold);

        clock_t begin = clock();

        //#############  Calling Arnoldi to get Q and H  ##################
        parallelArnoldi(d_A, d_q1, k, d_Q, d_H1, n, d_z, temp);


        //###############  Calling SVD on H-I   #####################

        signed char jobu = 'A'; // all m columns of U
        signed char jobvt = 'A'; // all n columns of VT
        cusolverDnDgesvd (
        cusolverH,
        jobu,
        jobvt,
        rows,
        cols,
        (double*)d_H1,
        lda,
        (double*)d_S,
        (double*)d_U,
        lda,
        (double*)d_VT,
        lda,
        (double*)d_work,
        lwork,
        (double*)d_rwork,
        devInfo);
        cudaDeviceSynchronize();

/*
        cudaMemcpy(U , d_U , sizeof(double)*lda*rows, cudaMemcpyDeviceToHost);
        cudaMemcpy(VT, d_VT, sizeof(double)*lda*cols, cudaMemcpyDeviceToHost);
        cudaMemcpy(S , d_S , sizeof(double)*cols , cudaMemcpyDeviceToHost);
        cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

        printf("S = (matlab base-1)\n");
        printMatrix(cols, 1, S, lda, "S");
        printf("=====\n");

        printf("U = (matlab base-1)\n");
        printMatrix(rows, rows, U, lda, "U");
        printf("=====\n");

        printf("VT = (matlab base-1)\n");
        printMatrix(cols, cols, VT, lda, "VT");
        printf("=====\n");
*/
        get_v<<<1, k>>>(d_VT, d_v, k);

        saxpy<<<n/NUM_THREADS + 1, NUM_THREADS, k*sizeof(double)>>>(d_Q, d_v, d_q1, n, k);

        calc_norm<<<n/NUM_THREADS + 1, NUM_THREADS>>>(d_A, d_q1, n, d_norm);
        check<<<1, 1>>>(d_norm, n, temp);

        cudaMemcpy(&tempNorm, temp, sizeof(double), cudaMemcpyDeviceToHost);
        printf("Aq - q norm for iteration %d is %.15lf\n", fold, tempNorm);

        clock_t end = clock();

        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        printf("Time for this iteration = %lf\n", elapsed_secs);

        if (err != cudaSuccess) printf("Error above: %s\n", cudaGetErrorString(err));


        /*if(normDiff < 0.000001){
            printf("Converged in %d folds.\n", fold);
            break;
        }*/
    }

    cudaMemcpy(h_q , d_q1 , sizeof(double)*n, cudaMemcpyDeviceToHost);

    cudaFree(d_norm);
    cudaFree(d_A);
    cudaFree(d_H1);
    cudaFree(d_Q);
    cudaFree(d_q1);
    cudaFree(d_z);
    cudaFree(d_v);
    cudaFree(d_rwork);
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_VT);
    cudaFree(d_work);

    // end fold
    //printf("Eigenvector : \n");
    for(int i=0;i < n;i++){
           //printf("%lf\n",abs(h_q[i]));
        }
    free(A.A[0]);
    free(A.A);
    free(h_A);
    free(h_q);

    return 0;
}
