
#include "intercept_temp.h"

void getDescriptor(const cudnnTensorDescriptor_t desc) {

	int ndims = 10;
	cudnnDataType_t* dtype = (cudnnDataType_t*)malloc(sizeof(cudnnDataType_t));
	int* nbdims = (int*)malloc(sizeof(int));
	int dimA[10] = {0};
	int strideA[10] = {0};

	cudnnStatus_t status = cudnnGetTensorNdDescriptor(desc, ndims, dtype, nbdims, dimA, strideA);
	printf("%d\n", *dtype);

	assert (status==CUDNN_STATUS_SUCCESS);

}

cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {


	int idx = get_idx();
	assert (idx >= 0);
	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

	DEBUG_PRINT("[INTERCEPTER-CATCH-%d]-[%d] Caught cudnnConvolutionForward, CUDNN handle is %p\n", idx, func_indexes[idx], handle, idx);

	if (idx < 2)
		block(idx,  mutexes, kqueues);

	// create record
	cudnnConvolutionForward_record new_conv_record = {
		handle,
		alpha,
		xDesc,
		x,
		wDesc,
		w,
		convDesc,
		algo,
		workSpace,
		workSpaceSizeInBytes,
		beta,
		yDesc,
		y
	};
	union func_data new_func_data;
	new_func_data.cudnnConvRecord = new_conv_record;
	func_record new_record = {CUDNN_CONV_RECORD, new_func_data};

	// push or run
	if (idx < 2) {
		 pthread_mutex_lock(mutexes[idx]);
		 kqueues[idx]->push(new_record);
		 pthread_mutex_unlock(mutexes[idx]);

		 func_indexes[idx] += 1;
		 block(idx, mutexes, kqueues);
	}
	else {

		cudnnStatus_t (*function)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) ;
		*(void **)(&function) = dlsym(RTLD_NEXT, "cudnnConvolutionForward");
		assert(function != NULL);

		status = (*function)(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
		if (status != CUDNN_STATUS_SUCCESS)
			printf("status is %d\n", status);
		assert (status == CUDNN_STATUS_SUCCESS);

		DEBUG_PRINT("CONV submitted!!\n");
	}

	return status;

}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc,  const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScaleData, const void *bnBiasData, double exponentialAverageFactor, void *resultRunningMeanData, void *resultRunningVarianceData, double epsilon, void *saveMean, void *saveInvVariance, const cudnnActivationDescriptor_t activationDesc,  void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) {

	int idx = get_idx();
	assert (idx >= 0);
	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

	DEBUG_PRINT("[INTERCEPTER-CATCH-%d]-[%d] Caught cudnnBatchNormalizationForwardTrainingEx, handle is %p\n", idx, func_indexes[idx], handle);

	if (idx < 2)
		block(idx,  mutexes, kqueues);

	// create record
	cudnnBatchNormalizationForwardTrainingEx_record new_bn_record = {
		handle,
		mode,
		bnOps,
		alpha,
		beta,
		xDesc,
		xData,
		zDesc,
		zData,
		yDesc,
		yData,
		bnScaleBiasMeanVarDesc,
		bnScaleData,
		bnBiasData,
		exponentialAverageFactor,
		resultRunningMeanData,
		resultRunningVarianceData,
		epsilon,
		saveMean,
		saveInvVariance,
		activationDesc,
		workspace,
		workSpaceSizeInBytes,
		reserveSpace,
		reserveSpaceSizeInBytes

	};
	union func_data new_func_data;
	new_func_data.cudnnBNormRecord = new_bn_record;
	func_record new_record = {CUDNN_BNORM_RECORD, new_func_data};

	// push or run

	if (idx < 2) {

		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);

		func_indexes[idx] += 1;
		block(idx,  mutexes, kqueues);
	}
	else {
		cudnnStatus_t (*function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc,  const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScaleData, const void *bnBiasData, double exponentialAverageFactor, void *resultRunningMeanData, void *resultRunningVarianceData, double epsilon, void *saveMean, void *saveInvVariance, const cudnnActivationDescriptor_t activationDesc,  void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes);

		*(void **)(&function) = dlsym(RTLD_NEXT, "cudnnBatchNormalizationForwardTrainingEx");
		assert(function != NULL);

		status = (*function)(handle, mode, bnOps, alpha, beta, xDesc, xData, zDesc, zData, yDesc, yData, bnScaleBiasMeanVarDesc, bnScaleData, bnBiasData, exponentialAverageFactor, resultRunningMeanData, resultRunningVarianceData, epsilon, saveMean, saveInvVariance, activationDesc, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
		if (status != CUDNN_STATUS_SUCCESS)
			printf("status is %d\n", status);
		assert (status == CUDNN_STATUS_SUCCESS);

		DEBUG_PRINT("BNORM submitted!!\n");

	}

	return status;
}


cudnnStatus_t cudnnBatchNormalizationForwardInference(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon)

{

	int idx = get_idx();
	assert (idx >= 0);
	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

	DEBUG_PRINT("[INTERCEPTER-CATCH]-[%d] Caught cudnnBatchNormalizationForwardInference, handle is %p, index is %d\n", func_indexes[idx], handle, idx);

	if (idx < 2)
		block(idx,  mutexes, kqueues);


	// create record
	cudnnBatchNormalizationForwardInference_record bn_record = {
		handle,
		mode,
		alpha,
		beta,
		xDesc,
		x,
		yDesc,
		y,
		bnScaleBiasMeanVarDesc,
		bnScale,
		bnBias,
		estimatedMean,
		estimatedVariance,
		epsilon
	};

	union func_data new_func_data;
	new_func_data.cudnnBNormInfRecord = bn_record;
	func_record new_record = {CUDNN_BNORM_INF_RECORD, new_func_data};

	if (idx < 2) {

		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);

		func_indexes[idx] += 1;
		block(idx,  mutexes, kqueues);

	}
	else {

		cudnnStatus_t (*function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon);

		*(void **)(&function) = dlsym(RTLD_NEXT, "cudnnBatchNormalizationForwardInference");
		assert(function != NULL);

		status = (*function)(handle, mode, alpha, beta, xDesc, x, xDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
		assert (status == CUDNN_STATUS_SUCCESS);

	}

	return status;
}


cudnnStatus_t cudnnRNNForwardInference(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes)  {

	int idx = get_idx();
	assert (idx >= 0);
	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;


	DEBUG_PRINT("[INTERCEPTER-CATCH]-[%d] Caught cudnnRNNForwardInference, handle is %p, index is %d\n", func_indexes[idx], handle, idx);
	printf("------------------------------------------------- IDX [%d], CX IS %p, CY IS %p\n", idx, cx, cy);

	if (idx < 2) {

		cudnnTensorDescriptor_t* xDesc_new = (cudnnTensorDescriptor_t*)malloc(sizeof(cudnnTensorDescriptor_t));
	        //cudnnStatus_t s = cudnnCreateTensorDescriptor(xDesc_new);

		*xDesc_new = *xDesc;
		//printf("%p, %p, %p, %p\n", xDesc, *xDesc, xDesc_new, *(xDesc_new));
		//memcpy(xDesc_new, xDesc, sizeof(cudnnTensorDescriptor_t));

		cudnnTensorDescriptor_t* yDesc_new = (cudnnTensorDescriptor_t*)malloc(sizeof(cudnnTensorDescriptor_t));
		*yDesc_new = *yDesc;


		cudnnRNNForwardInf_record rnn_record = {
			handle,
			rnnDesc,
			seqLength,
			xDesc_new,
			x,
			hxDesc,
			hx,
			cxDesc,
			cx,
			wDesc,
			w,
			yDesc_new,
			y,
			hyDesc,
			hy,
			cyDesc,
			cy,
			workspace,
			workSpaceSizeInBytes
		};

		union func_data new_func_data;
		new_func_data.cudnnRnnInfRecord = rnn_record;
		func_record new_record = {CUDNN_RNN_INF_RECORD, new_func_data};

		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);

		func_indexes[idx] += 1;
		block(idx,  mutexes, kqueues);
	}
	else {
		cudnnStatus_t (*function)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes);

		*(void **)(&function) = dlsym(RTLD_NEXT, "cudnnRNNForwardInference");
		assert(function != NULL);

		status = (*function)(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes);


		printf("------------------------- cudnn status is %d\n", status);
		// TODO: not sure why this complains here in just one call!
		assert (status == CUDNN_STATUS_SUCCESS);

		// cudaError_t err_all = cudaDeviceSynchronize(); // for debugging
		// CHECK_CUDA_ERROR(err_all);
	}

	return status;

}


cudnnStatus_t cudnnRNNForwardTraining(
	cudnnHandle_t handle,
	const cudnnRNNDescriptor_t rnnDesc,
	const int seqLength,
	const cudnnTensorDescriptor_t *xDesc,
	const void *x,
    const cudnnTensorDescriptor_t hxDesc,
    const void *hx,
    const cudnnTensorDescriptor_t cxDesc,
    const void *cx,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t *yDesc,
    void *y,
    const cudnnTensorDescriptor_t hyDesc,
    void *hy,
    const cudnnTensorDescriptor_t cyDesc,
    void *cy,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {

	int idx = get_idx();
	assert (idx >= 0);
	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

	DEBUG_PRINT("[INTERCEPTER-CATCH]-[%d] Caught cudnnRNNForwardTraining, handle is %p, index is %d\n", func_indexes[idx], handle, idx);

	if (idx < 2) {

		cudnnTensorDescriptor_t* xDesc_new = (cudnnTensorDescriptor_t*)malloc(sizeof(cudnnTensorDescriptor_t));
	        //cudnnStatus_t s = cudnnCreateTensorDescriptor(xDesc_new);

		*xDesc_new = *xDesc;
		printf("%p, %p, %p, %p\n", xDesc, *xDesc, xDesc_new, *(xDesc_new));
		//memcpy(xDesc_new, xDesc, sizeof(cudnnTensorDescriptor_t));

		cudnnTensorDescriptor_t* yDesc_new = (cudnnTensorDescriptor_t*)malloc(sizeof(cudnnTensorDescriptor_t));
		*yDesc_new = *yDesc;

		cudnnRNNForwardTraining_record rnn_record = {
			handle,
			rnnDesc,
			seqLength,
			xDesc,
			x,
			hxDesc,
			hx,
			cxDesc,
			cx,
			wDesc,
			w,
			yDesc,
			y,
			hyDesc,
			hy,
			cyDesc,
			cy,
			workspace,
			workSpaceSizeInBytes,
			reserveSpace,
			reserveSpaceSizeInBytes
		};

		union func_data new_func_data;
		new_func_data.cudnnRnnTrainRecord = rnn_record;
		func_record new_record = {CUDNN_RNN_TRAIN_RECORD, new_func_data};

		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);

		func_indexes[idx] += 1;
		block(idx,  mutexes, kqueues);
	}
	else {

		cudnnStatus_t (*function)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes);
		*(void **)(&function) = dlsym(RTLD_NEXT, "cudnnRNNForwardTraining");
		assert(function != NULL);

		status = (*function)(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

		// cudaError_t err_all = cudaDeviceSynchronize(); // for debugging
		// CHECK_CUDA_ERROR(err_all);
	}

	return status;
}

// cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
// 	cudnnHandle_t handle,
// 	cudnnBatchNormMode_t mode,
// 	cudnnBatchNormOps_t bnOps,
// 	const cudnnActivationDescriptor_t activationDesc,
// 	const cudnnTensorDescriptor_t xDesc,
//     size_t *sizeInBytes
// ) {

// 	int idx = get_idx();
// 	assert (idx >= 0);
// 	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

// 	DEBUG_PRINT("[INTERCEPTER-CATCH]-[%d] Caught cudnnGetBatchNormalizationTrainingExReserveSpaceSize, handle is %p, index is %d\n", func_indexes[idx], handle, idx);


// 	if (idx < 2) {

// 		cudnnBatchNormReserve_record cr_record = {handle, mode, bnOps, activationDesc, xDesc, sizeInBytes};
// 		union func_data new_func_data;
// 		new_func_data.cudnnBNormResRecord = cr_record;
// 		func_record new_record = {CUDNN_BNORM_RESERVE_RECORD, new_func_data};

// 		pthread_mutex_lock(mutexes[idx]);
// 		kqueues[idx]->push(new_record);
// 		pthread_mutex_unlock(mutexes[idx]);

// 		block(idx,  mutexes, kqueues);
// 	}
// 	else {
// 		cudnnStatus_t (*function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes);
// 		*(void **)(&function) = dlsym(RTLD_NEXT, "cudnnGetBatchNormalizationTrainingExReserveSpaceSize");
// 		assert(function != NULL);

// 		status = (*function)(handle, mode, bnOps, activationDesc, xDesc, sizeInBytes);
// 		if (status != CUDNN_STATUS_SUCCESS)
// 			printf("status is %d\n", status);
// 		assert (status == CUDNN_STATUS_SUCCESS);

// 		//CHECK_CUDA_ERROR(cudaDeviceSynchronize());
// 	}

// 	return status;

// }

cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) {

	//DEBUG_PRINT("Caught a cudnnDestroyRNNDescriptor! Do nothing!\n");
	return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {

	// mock cudnn destroy TensorDescriptor
	//DEBUG_PRINT("Caught a cudnnDestroyTensorDescriptor! Do nothing!\n");
	return CUDNN_STATUS_SUCCESS;
}


cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {

	//DEBUG_PRINT("Caught a cudnnDestroyFilterDescriptor! Do nothing!\n");
	return CUDNN_STATUS_SUCCESS;

}


cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {

	//DEBUG_PRINT("Caught a cudnnDestroyConvolutionDescriptor! Do nothing!\n");
	return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc) {
	//DEBUG_PRINT("Caught a cudnnDestroyDropoutDescriptor! Do nothing!\n");
	return CUDNN_STATUS_SUCCESS;

}



// cudnnStatus_t cudnnCreate(cudnnHandle_t *handle) {

// 	int idx = get_idx();
// 	assert (idx >= 0);
// 	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

// 	printf("Caught a cudnnCreate!\n");

// 	// cudnnCreate_record cr_record = {handle};
// 	// union func_data new_func_data;
// 	// new_func_data.cudnnCreateRecord = cr_record;
// 	// func_record new_record = {CUDNN_CREATE_RECORD, new_func_data};

// 	cudnnStatus_t (*function)(cudnnHandle_t *handle);
// 	*(void **)(&function) = dlsym(RTLD_NEXT, "cudnnCreate");
// 	assert(function != NULL);

// 	status = (*function)(handle);
// 	assert (status == CUDNN_STATUS_SUCCESS);

// 	status = cudnnSetStream(*handle, client_streams[idx]);
// 	if (status != CUDNN_STATUS_SUCCESS)
// 		printf("status is %d\n", status);
// 	assert (status == CUDNN_STATUS_SUCCESS);

// 	// if (idx < 2) {

// 	// 	pthread_mutex_lock(mutexes[idx]);
// 	// 	kqueues[idx]->push(new_record);
// 	// 	pthread_mutex_unlock(mutexes[idx]);

// 	// 	block(idx,  mutexes, kqueues);
// 	// }
// 	// else {
// 	// 	cudnnStatus_t (*function)(cudnnHandle_t *handle);
// 	// 	*(void **)(&function) = dlsym(RTLD_NEXT, "cudnnCreate");
// 	// 	assert(function != NULL);

// 	// 	status = (*function)(handle);
// 	// 	assert (status == CUDNN_STATUS_SUCCESS);
// 	// }

// 	return status;
// }


// cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId) {
// 	int idx = get_idx();
// 	assert (idx >= 0);

// 	//printf("Caught a cudnnSetStream from idx %d, stream is %d!\n", idx, streamId);

// 	if (idx < 2)
// 		return CUDNN_STATUS_SUCCESS;

// 	cudnnStatus_t (*function)(cudnnHandle_t handle, cudaStream_t streamId);
// 	*(void **)(&function) = dlsym(RTLD_NEXT, "cudnnSetStream");
// 	assert(function != NULL);

// 	cudnnStatus_t status = (*function)(handle, streamId);
// 	//cudaError_t err_all = cudaDeviceSynchronize(); // for debugging
// 	//CHECK_CUDA_ERROR(err_all);
// 	return status;
// }


cudnnStatus_t cudnnDestroy(cudnnHandle_t handle) {

	printf("Caught a cudnnDestroy, Do nothing!\n ");
	return CUDNN_STATUS_SUCCESS;
}