/* Intercepts and overwrites CUBLAS calls */

#include "intercept_temp.h"

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {

	int idx = get_idx();
	assert (idx >= 0);
	cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

	cublasSgemm_record blassgemm_record = {
		handle,
		transa,
		transb,
		m,
		n,
		k,
		alpha,
		A,
		lda,
		B,
		ldb,
		beta,
		C,
		ldc
	};

	union func_data new_func_data;
	new_func_data.cublasSgemmRecord = blassgemm_record;
	func_record new_record = {CUBLAS_SGEMM_RECORD, new_func_data};

	if (idx < *num_total_clients) {

		pthread_mutex_lock(mutexes[idx]);
		DEBUG_PRINT("[INTERCEPTER-CATCH]-[%d] Caught cublasSgemm_v2, handle is %p, index %d, m is %d, n is %d, k is %d\n", func_indexes[idx], handle, idx, m, n, k);
		kqueues[idx]->push(new_record);
		func_indexes[idx] += 1;
		pthread_mutex_unlock(mutexes[idx]);

		block(idx,  mutexes, kqueues);
	}
	else {

		if (cublas_sgemm_func==NULL) {
			*(void **)(&cublas_sgemm_func) = dlsym(RTLD_NEXT, "cublasSgemm_v2");
			assert(cublas_sgemm_func != NULL);
		}
		status = (*cublas_sgemm_func)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		assert (status == CUBLAS_STATUS_SUCCESS);
		DEBUG_PRINT("CUBLAS status is %d\n", status);

	}

	return status;

}



cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {

	int idx = get_idx();
	assert (idx >= 0);
	cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

	cublasSgemm_record blassgemm_record = {
		handle,
		transa,
		transb,
		m,
		n,
		k,
		alpha,
		A,
		lda,
		B,
		ldb,
		beta,
		C,
		ldc
	};

	union func_data new_func_data;
	new_func_data.cublasSgemmRecord = blassgemm_record;
	func_record new_record = {CUBLAS_SGEMM_RECORD, new_func_data};

	if (idx < *num_total_clients) {

		pthread_mutex_lock(mutexes[idx]);
		DEBUG_PRINT("[INTERCEPTER-CATCH]-[%d] Caught cublasSgemm, handle is %p, index %d, m is %d, n is %d, k is %d\n", func_indexes[idx], handle, idx, m, n, k);
		kqueues[idx]->push(new_record);
		func_indexes[idx] += 1;
		pthread_mutex_unlock(mutexes[idx]);

		block(idx,  mutexes, kqueues);
	}
	else {

		if (cublas_sgemm_func==NULL) {
			*(void **)(&cublas_sgemm_func) = dlsym(RTLD_NEXT, "cublasSgemm_v2");
			assert(cublas_sgemm_func != NULL);
		}
		status = (*cublas_sgemm_func)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		assert (status == CUBLAS_STATUS_SUCCESS);
		DEBUG_PRINT("CUBLAS status is %d\n", status);

	}

	return status;

}


cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount) {

	int idx = get_idx();
	assert (idx >= 0);
	cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

	cublasSgemmStridedBatched_record record = {
		handle,
		transa,
		transb,
		m,
		n,
		k,
		alpha,
		A,
		lda,
		strideA,
		B,
		ldb,
		strideB,
		beta,
		C,
		ldc,
		strideC,
		batchCount
	};

	union func_data new_func_data;
	new_func_data.cublasSgemmStridedRecord = record;
	func_record new_record = {CUBLAS_SGEMM_STRIDED_RECORD, new_func_data};

	if (idx < *num_total_clients) {

		pthread_mutex_lock(mutexes[idx]);
		DEBUG_PRINT("[INTERCEPTER-CATCH]-[%d] Caught cublasSgemmStridedBatched, handle is %p\n", func_indexes[idx], handle);
		kqueues[idx]->push(new_record);
		func_indexes[idx] += 1;
		pthread_mutex_unlock(mutexes[idx]);

		block(idx,  mutexes, kqueues);

	}
	else {

		if (cublas_sgemm_strided_func==NULL) {
			*(void **)(&cublas_sgemm_strided_func) = dlsym(RTLD_NEXT, "cublasSgemmStridedBatched");
			assert(cublas_sgemm_strided_func != NULL);
		}

		status = (*cublas_sgemm_strided_func)(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
		assert (status == CUBLAS_STATUS_SUCCESS);
		DEBUG_PRINT("CUBLAS status is %d\n", status);

	}

	return status;
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {

	DEBUG_PRINT("Caught a cublasDestroy! Do nothing!\n");
	return CUBLAS_STATUS_SUCCESS;
}
