
#include "intercept_temp.h"

cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {


	int idx = get_idx();
	assert (idx >= 0);
	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

	if (cudnn_conv_func==NULL) {
		*(void **)(&cudnn_conv_func) = dlsym(RTLD_NEXT, "cudnnConvolutionForward");
		assert(cudnn_conv_func != NULL);
	}

	std::pair<cudaStream_t, cudaEvent_t> sched_pair = push_and_wait(CUDNN_CONV_RECORD, true);
	wait_events(sched_pair.first);
	cudnnSetStream(handle, sched_pair.first);

	status = (*cudnn_conv_func)(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
	if (status != CUDNN_STATUS_SUCCESS)
		printf("status is %d\n", status);
	assert (status == CUDNN_STATUS_SUCCESS);
	DEBUG_PRINT("CONV submitted!!\n");
	CHECK_CUDA_ERROR(cudaEventRecord(sched_pair.second, sched_pair.first));

	return status;

}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc,  const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScaleData, const void *bnBiasData, double exponentialAverageFactor, void *resultRunningMeanData, void *resultRunningVarianceData, double epsilon, void *saveMean, void *saveInvVariance, const cudnnActivationDescriptor_t activationDesc,  void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) {

	int idx = get_idx();
	assert (idx >= 0);
	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

	if (cudnn_bnorm_func==NULL) {
		*(void **)(&cudnn_bnorm_func) = dlsym(RTLD_NEXT, "cudnnBatchNormalizationForwardTrainingEx");
		assert(cudnn_bnorm_func != NULL);
	}

	std::pair<cudaStream_t, cudaEvent_t> sched_pair = push_and_wait(CUDNN_BNORM_RECORD, true);
	wait_events(sched_pair.first);
	cudnnSetStream(handle, sched_pair.first);

	//printf("run func %p\n", cudnn_bnorm_function);
	status = (*cudnn_bnorm_func)(handle, mode, bnOps, alpha, beta, xDesc, xData, zDesc, zData, yDesc, yData, bnScaleBiasMeanVarDesc, bnScaleData, bnBiasData, exponentialAverageFactor, resultRunningMeanData, resultRunningVarianceData, epsilon, saveMean, saveInvVariance, activationDesc, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
	assert (status == CUDNN_STATUS_SUCCESS);
	CHECK_CUDA_ERROR(cudaEventRecord(sched_pair.second, sched_pair.first));

	return status;
}


cudnnStatus_t cudnnBatchNormalizationForwardInference(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon)

{

	int idx = get_idx();
	assert (idx >= 0);
	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
	// if (idx < 2)
	// 	block(idx,  mutexes, kqueues);

	if (cudnn_bnorm_infer_func==NULL) {
		*(void **)(&cudnn_bnorm_infer_func) = dlsym(RTLD_NEXT, "cudnnBatchNormalizationForwardInference");
		assert(cudnn_bnorm_infer_func != NULL);
	}

	std::pair<cudaStream_t, cudaEvent_t> sched_pair = push_and_wait(CUDNN_BNORM_INF_RECORD, true);
	wait_events(sched_pair.first);
	cudnnSetStream(handle, sched_pair.first);

	status = (*cudnn_bnorm_infer_func)(handle, mode, alpha, beta, xDesc, x, xDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
	assert (status == CUDNN_STATUS_SUCCESS);
	CHECK_CUDA_ERROR(cudaEventRecord(sched_pair.second, sched_pair.first));

	return status;
}


cudnnStatus_t cudnnRNNForwardInference(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes)  {

	int idx = get_idx();
	assert (idx >= 0);
	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

	if (cudnn_rnn_func==NULL) {
		*(void **)(&cudnn_rnn_func) = dlsym(RTLD_NEXT, "cudnnRNNForwardInference");
		assert(cudnn_rnn_func != NULL);
	}

	std::pair<cudaStream_t, cudaEvent_t> sched_pair = push_and_wait(CUDNN_RNN_INF_RECORD, true);
	wait_events(sched_pair.first);
	cudnnSetStream(handle, sched_pair.first);

	status = (*cudnn_rnn_func)(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes);
	// TODO: not sure why this complains here in just one call!
	assert (status == CUDNN_STATUS_SUCCESS);
	CHECK_CUDA_ERROR(cudaEventRecord(sched_pair.second, sched_pair.first));

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

	if (cudnn_rnn_train_func==NULL) {
		*(void **)(&cudnn_rnn_train_func) = dlsym(RTLD_NEXT, "cudnnRNNForwardTraining");
		assert(cudnn_rnn_train_func != NULL);
	}

	std::pair<cudaStream_t, cudaEvent_t> sched_pair = push_and_wait(CUDNN_RNN_TRAIN_RECORD, true);
	wait_events(sched_pair.first);
	cudnnSetStream(handle, sched_pair.first);

	status = (*cudnn_rnn_train_func)(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
	assert (status == CUDNN_STATUS_SUCCESS);
	CHECK_CUDA_ERROR(cudaEventRecord(sched_pair.second, sched_pair.first));

	return status;
}


// backward

cudnnStatus_t cudnnBatchNormalizationBackwardEx (
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const void *alphaDataDiff,
    const void *betaDataDiff,
    const void *alphaParamDiff,
    const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t yDesc,
    const void *yData,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dyData,
    const cudnnTensorDescriptor_t dzDesc,
    void *dzData,
    const cudnnTensorDescriptor_t dxDesc,
    void *dxData,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScaleData,
    const void *bnBiasData,
    void *dBnScaleData,
    void *dBnBiasData,
    double epsilon,
    const void *savedMean,
    const void *savedInvVariance,
    const cudnnActivationDescriptor_t activationDesc,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
) {

	int idx = get_idx();
	assert (idx >= 0);

	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

	if (cudnn_bnorm_bw_func==NULL) {
		*(void **)(&cudnn_bnorm_bw_func) = dlsym(RTLD_NEXT, "cudnnBatchNormalizationBackwardEx");
		assert(cudnn_bnorm_bw_func != NULL);
	}

	status = (*cudnn_bnorm_bw_func)(
			handle,
			mode,
			bnOps,
			alphaDataDiff,
			betaDataDiff,
			alphaParamDiff,
			betaParamDiff,
			xDesc,
			xData,
			yDesc,
			yData,
			dyDesc,
			dyData,
			dzDesc,
			dzData,
			dxDesc,
			dxData,
			dBnScaleBiasDesc,
			bnScaleData,
			bnBiasData,
			dBnScaleData,
			dBnBiasData,
			epsilon,
			savedMean,
			savedInvVariance,
			activationDesc,
			workspace,
			workSpaceSizeInBytes,
			reserveSpace,
			reserveSpaceSizeInBytes
	);

	std::pair<cudaStream_t, cudaEvent_t> sched_pair = push_and_wait(CUDNN_BNORM_BACKWARD_RECORD, true);
	wait_events(sched_pair.first);
	cudnnSetStream(handle, sched_pair.first);

	assert (status == CUDNN_STATUS_SUCCESS);
	CHECK_CUDA_ERROR(cudaEventRecord(sched_pair.second, sched_pair.first));

	return status;
}

cudnnStatus_t cudnnConvolutionBackwardData(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx
)
{

	int idx = get_idx();
	assert (idx >= 0);
	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

	if (cudnn_conv_bw_data_func==NULL) {
		*(void **)(&cudnn_conv_bw_data_func) = dlsym(RTLD_NEXT, "cudnnConvolutionBackwardData");
		assert(cudnn_conv_bw_data_func != NULL);
	}
	status = (*cudnn_conv_bw_data_func)(
			handle,
			alpha,
			wDesc,
			w,
			dyDesc,
			dy,
			convDesc,
			algo,
			workSpace,
			workSpaceSizeInBytes,
			beta,
			dxDesc,
			dx
	);

	std::pair<cudaStream_t, cudaEvent_t> sched_pair = push_and_wait(CUDNN_CONV_DATA_RECORD, true);
	wait_events(sched_pair.first);
	cudnnSetStream(handle, sched_pair.first);

	assert (status == CUDNN_STATUS_SUCCESS);
	CHECK_CUDA_ERROR(cudaEventRecord(sched_pair.second, sched_pair.first));
	return status;
}

cudnnStatus_t cudnnConvolutionBackwardFilter(
    cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *beta,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw
) {

	int idx = get_idx();
	assert (idx >= 0);

	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

	if (cudnn_conv_bw_filter_func==NULL) {
		*(void **)(&cudnn_conv_bw_filter_func) = dlsym(RTLD_NEXT, "cudnnConvolutionBackwardFilter");
		assert(cudnn_conv_bw_filter_func != NULL);
	}

	status = (*cudnn_conv_bw_filter_func)(
			handle,
			alpha,
			xDesc,
			x,
			dyDesc,
			dy,
			convDesc,
			algo,
			workSpace,
			workSpaceSizeInBytes,
			beta,
			dwDesc,
			dw
	);

	std::pair<cudaStream_t, cudaEvent_t> sched_pair = push_and_wait(CUDNN_CONV_FILTER_RECORD, true);
	wait_events(sched_pair.first);
	cudnnSetStream(handle, sched_pair.first);

	assert (status == CUDNN_STATUS_SUCCESS);
	CHECK_CUDA_ERROR(cudaEventRecord(sched_pair.second, sched_pair.first));

	return status;
}


cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc) {
	return CUDNN_STATUS_SUCCESS;
}

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


cudnnStatus_t cudnnDestroy(cudnnHandle_t handle) {

	printf("Caught a cudnnDestroy, Do nothing!\n ");
	return CUDNN_STATUS_SUCCESS;
}