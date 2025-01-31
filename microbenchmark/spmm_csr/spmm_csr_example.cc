/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
// #include "cublas_utils.h"

#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM

#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <iostream>
#include <getopt.h>
// #include <cstdlib>

#define CHECK_CUBLAS(func)                                                       \
{                                                                              \
    cublasStatus_t status = (func);                                               \
    if (status != CUBLAS_STATUS_SUCCESS) {                                               \
        printf("CUBLAS API failed at line %d with error: %d\n",             \
               __LINE__, status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

// Function to initialize a matrix with random data
void initMatrix(float* mat, int numRows, int numCols) {
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            mat[i * numCols + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
}

void printMatrix(float* mat, int numRows, int numCols) {
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            std::cout << mat[i * numCols + j] << std::endl;
        }
    }
}

void initMatrix(float **mat, int numRows, int numCols) {
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            mat[i][j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
}


// AB (A weight, B activation)
// A (M x K)
// B (K x N)
float run_gemm(
    const int A_num_rows, // M
    const int A_num_cols, // K
    const int B_num_cols, // N
    int num_tests=10000
) {

    const int M = A_num_rows;
    const int N = A_num_cols;
    const int K = B_num_cols;

    int   A_size          = M * K;
    int   B_size          = K * N;
    int   C_size          = M * K;

    float *hA = new float[A_size];
    float *hB = new float[B_size];
    float *hC = new float[C_size];

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    // Initialize dense A & B
    {
        initMatrix(hA, A_size, 1);
        initMatrix(hB, B_size, 1);
        initMatrix(hC, C_size, 1);

        // printMatrix(hB, B_size, 1);
        // std::cout << "print finished for Mat B" << std::endl;
        // printMatrix(hC, C_size, 1);
        // std::cout << "print finished for Mat C" << std::endl;
    }

    //--------------------------------------------------------------------------
    // Device memory management
    float *dA, *dB, *dC;

    CHECK_CUDA( cudaMalloc((void**) &dA, A_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(float),
                           cudaMemcpyHostToDevice) )

    const float alpha           = 1.0f;
    const float beta            = 0.0f;

    //--------------------------------------------------------------------------
    // CUBLAS APIs
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    CHECK_CUBLAS(cublasCreate(&cublasH));
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUBLAS(cublasSetStream(cublasH, stream));

    ////////////////////////////////////////////////////////////////////////////
    // RUN TEST
    ////////////////////////////////////////////////////////////////////////////

    // warmup
    for (int i = 0; i < 20; i++){
        CHECK_CUBLAS(
            cublasSgemm(cublasH,
                transa,
                transb,
                // M, N, K,
                N, M, K,
                &alpha,
                dB, N,
                dA, K,
                &beta,
                dC, N
            )
        );
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Starting a range
    nvtxRangePush("CUBLAS_test");

    // Record the start event
    CHECK_CUDA(cudaEventRecord(start));

    // execute SpMM
    for (int i = 0; i < num_tests; i++){
        CHECK_CUBLAS(
            cublasSgemm(cublasH,
                transa,
                transb,
                // M, N, K,
                N, M, K,
                &alpha,
                dB, N,
                dA, K,
                &beta,
                dC, N
            )
        );
    }

    // Record the end event
    CHECK_CUDA(cudaEventRecord(stop));

    // Wait for the stop event to complete
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate the elapsed time between the start and stop events
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Ending a range
    nvtxRangePop();


    /////////////////////////////////////////////////////
    // CLEANUP
    /////////////////////////////////////////////////////
    // device memory
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    
    // host mem
    free(hA);
    free(hB);
    free(hC);

    // event, handle, stream, device
    CHECK_CUDA(cudaEventDestroy(start))
    CHECK_CUDA(cudaEventDestroy(stop))
    CHECK_CUBLAS(cublasDestroy(cublasH));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaDeviceReset());

    return milliseconds;
}


float run_spmm(
    const float sparsity,
    const int A_num_rows,
    const int A_num_cols,
    const int B_num_cols,
    int num_tests=10000
) {

    const int   A_nnz           = (1-sparsity) * A_num_rows * A_num_cols;
    const int   B_num_rows      = A_num_cols;

    int   ldb             = B_num_rows;
    int   ldc             = A_num_rows;
    int   B_size          = ldb * B_num_cols;
    int   C_size          = ldc * B_num_cols;

    float *hA_values = new float[A_nnz];
    int   *hA_csrOffsets = new int[A_num_rows + 1];
    int   *hA_columns = new int[A_nnz];
    float *hB = new float[B_size];
    float *hC = new float[C_size];

    // Initialize sparse A
    {
        // Initialize values
        initMatrix(hA_values, A_nnz, 1);

        // Initialize the column indices with random indices (for demonstration)
        for (int i = 0; i < A_nnz; ++i) {
            hA_columns[i] = rand() % A_num_cols;
        }

        // Initialize the row pointers in a way that the non-zeros are roughly evenly distributed
        // (This is just for demonstration purposes)
        hA_csrOffsets[0] = 0;
        for (int i = 1; i <= A_num_rows; ++i) {
            hA_csrOffsets[i] = hA_csrOffsets[i - 1] + (A_nnz / A_num_rows);
        }
    }

    // Initialize dense B & C
    {
        initMatrix(hB, B_size, 1);
        initMatrix(hC, C_size, 1);
    }

    float alpha           = 1.0f;
    float beta            = 0.0f;

    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values;
    float *dB, *dC;

    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets, (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))  )

    CHECK_CUDA( cudaMalloc((void**) &dB,         B_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC,         C_size * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    
    ////////////////////////////////////////////////////////////////////////////
    // RUN TEST
    ////////////////////////////////////////////////////////////////////////////
    float milliseconds = 0;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Starting a range
    nvtxRangePush("SpMM_test");

    // Record the start event
    CHECK_CUDA(cudaEventRecord(start));

    // execute SpMM
    for (int i = 0; i < num_tests; i++){
        CHECK_CUSPARSE( cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )
    }

    // Record the end event
    CHECK_CUDA(cudaEventRecord(stop));

    // Wait for the stop event to complete
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate the elapsed time between the start and stop events
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Ending a range
    nvtxRangePop();

    /////////////////////////////////////////////////////
    // CLEANUP
    /////////////////////////////////////////////////////

    // matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    // device mem
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    
    // host mem
    free(hA_values);
    free(hA_csrOffsets);
    free(hA_columns);
    free(hB);
    free(hC);

    // event, device
    CHECK_CUDA( cudaEventDestroy(start))
    CHECK_CUDA( cudaEventDestroy(stop))
    CHECK_CUDA(cudaDeviceReset());
    
    return milliseconds;
}

int N = 2048; // num_tokens
int K = 768; // dmodel for FFN1
int M = K * 4; // dff for FFN1
float sparsity = 0.99;

static void parse_opt(int argc, char**argv){
    int c;
    while((c = getopt(argc, argv, "m:k:n:s:")) != -1){
        switch(c){
            case 'm':
                M=atoi(optarg);
                break;
            case 'k':
                K=atoi(optarg);
                break;
            case 'n':
                N=atoi(optarg);
                break;
            case 's':
                sparsity=atof(optarg);
                break;
            default :
                printf("Wrong arg");
                exit(0);
        }
    }
}

int main(int argc, char**argv) {

    parse_opt(argc, argv);

    int num_tests = 1000;
    float milliseconds_dense = 0.0, milliseconds_sparse = 0.0;

    milliseconds_dense = run_gemm(M, K, N, num_tests) / num_tests;
    milliseconds_sparse = run_spmm(sparsity, M, K, N, num_tests) / num_tests;
    
    printf("[M,N,K,Sp]=[%d,%d,%d,%.1f] dense %f sparse %f\n", M,N,K,sparsity,
                                                    milliseconds_dense,
                                                    milliseconds_sparse);
    // std::cout << "[" << N << "] " \
    //             "DMM: " << milliseconds_dense << \
    //             " milliseconds // " << \
    //             " SpMM: " << milliseconds_sparse << \
    //             " milliseconds" << std::endl;

    return EXIT_SUCCESS;
}
