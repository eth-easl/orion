
#include "intercept_temp.h"

cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {

	int idx = get_idx();
	assert (idx >= 0);
	cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

	if (cublas_sgemm_func==NULL) {
		*(void **)(&cublas_sgemm_func) = dlsym(RTLD_NEXT, "cublasSgemm_v2");
		assert(cublas_sgemm_func != NULL);
	}

	cudaStream_t sched_stream = push_and_wait(CUBLAS_SGEMM_RECORD, true);
	cublasSetStream_v2(handle, sched_stream);

	status = (*cublas_sgemm_func)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	assert (status == CUBLAS_STATUS_SUCCESS);
	DEBUG_PRINT("CUBLAS status is %d\n", status);
	return status;

}


cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount) {

	int idx = get_idx();
	assert (idx >= 0);
	cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

	if (cublas_sgemm_strided_func==NULL) {
		*(void **)(&cublas_sgemm_strided_func) = dlsym(RTLD_NEXT, "cublasSgemmStridedBatched");
		assert(cublas_sgemm_strided_func != NULL);
	}

	cudaStream_t sched_stream = push_and_wait(CUBLAS_SGEMM_STRIDED_RECORD, true);
	cublasSetStream_v2(handle, sched_stream);

	status = (*cublas_sgemm_strided_func)(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
	assert (status == CUBLAS_STATUS_SUCCESS);
	DEBUG_PRINT("CUBLAS status is %d\n", status);
	return status;
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {

	DEBUG_PRINT("Caught a cublasDestroy! Do nothing!\n");
	return CUBLAS_STATUS_SUCCESS;
}
