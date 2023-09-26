#include <dlfcn.h>
#include <stdio.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include <queue>
#include <string>
#include <pthread.h>
#include <cuda.h>
#include <cudnn.h>
#include <cublas.h>
#include <assert.h>
#include <vector>
#include <sys/types.h>
#include <syscall.h>
#include <unistd.h>
#include <sys/syscall.h>
#include "../system_utils.h"

typedef struct kernel_record {

	const void* func;
	dim3 gridDim;
	dim3 blockDim;
	void** args;
	size_t sharedMem;
	cudaStream_t stream;
	volatile bool run;
	volatile cudaStream_t sched_stream;
} kernel_record;


typedef struct memcpy_record {

	void* dst;
	const void* src;
	size_t count;
	enum cudaMemcpyKind kind;
	cudaStream_t stream;
	bool async;
} memcpy_record;


typedef struct malloc_record {

	void** devPtr;
    size_t size;

} malloc_record;

typedef struct free_record {
	void* devPtr;

} free_record;

typedef struct memset_record {

	void* devPtr;
	int value;
	size_t count;
	cudaStream_t stream;
	bool async;

} memset_record;

// CUDNN

typedef struct cudnnConvolutionForward_record {

	cudnnHandle_t handle;
	const void *alpha;
	cudnnTensorDescriptor_t xDesc;
	const void *x;
	cudnnFilterDescriptor_t wDesc;
  	const void *w;
	cudnnConvolutionDescriptor_t convDesc;
	cudnnConvolutionFwdAlgo_t algo;
	void *workSpace;
	size_t workSpaceSizeInBytes;
	const void *beta;
	cudnnTensorDescriptor_t yDesc;
	void *y;

	cudnnConvolutionForward_record(cudnnHandle_t handle_arg, const void *alpha_arg, const cudnnTensorDescriptor_t xDesc_arg, const void *x_arg, const cudnnFilterDescriptor_t wDesc_arg, const void *w_arg, const cudnnConvolutionDescriptor_t convDesc_arg, cudnnConvolutionFwdAlgo_t algo_arg, void *workSpace_arg, size_t workSpaceSizeInBytes_arg, const void *beta_arg, const cudnnTensorDescriptor_t yDesc_arg, void *y_arg) {

		handle = handle_arg;
		alpha = alpha_arg;
		xDesc = xDesc_arg;
		x = x_arg;
		wDesc = wDesc_arg;
		w = w_arg;
		convDesc = convDesc_arg;
		algo = algo_arg;
		workSpace = workSpace_arg;
		workSpaceSizeInBytes = workSpaceSizeInBytes_arg;
		beta = beta_arg;
		yDesc = yDesc_arg;
		y = y_arg;
	};
	~cudnnConvolutionForward_record() {}


} cudnnConvolutionForward_record;


typedef struct cudnnBatchNormalizationForwardTrainingEx_record {

	cudnnHandle_t handle;
	cudnnBatchNormMode_t mode;
	cudnnBatchNormOps_t bnOps;
	const void *alpha;
	const void *beta;
	cudnnTensorDescriptor_t xDesc;
	const void *xData;
	cudnnTensorDescriptor_t zDesc;
	const void *zData;
  	cudnnTensorDescriptor_t yDesc;
	void  *yData;
	cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
	const void *bnScaleData;
	const void *bnBiasData;
	double exponentialAverageFactor;
	void  *resultRunningMeanData;
	void  *resultRunningVarianceData;
	double epsilon;
	void *saveMean;
	void *saveInvVariance;
	cudnnActivationDescriptor_t activationDesc;
	void *workspace;
	size_t workSpaceSizeInBytes;
	void *reserveSpace;
	size_t reserveSpaceSizeInBytes;

	cudnnBatchNormalizationForwardTrainingEx_record(cudnnHandle_t handle_arg, cudnnBatchNormMode_t mode_arg, cudnnBatchNormOps_t bnOps_arg, const void *alpha_arg, const void *beta_arg, const cudnnTensorDescriptor_t xDesc_arg, const void *xData_arg, const cudnnTensorDescriptor_t zDesc_arg,  const void *zData_arg, const cudnnTensorDescriptor_t yDesc_arg, void *yData_arg, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc_arg, const void *bnScaleData_arg, const void *bnBiasData_arg, double exponentialAverageFactor_arg, void *resultRunningMeanData_arg, void *resultRunningVarianceData_arg, double epsilon_arg, void *saveMean_arg, void *saveInvVariance_arg, const cudnnActivationDescriptor_t activationDesc_arg,  void *workspace_arg, size_t workSpaceSizeInBytes_arg, void *reserveSpace_arg, size_t reserveSpaceSizeInBytes_arg) {

		handle = handle_arg;
		mode = mode_arg;
		bnOps = bnOps_arg;
		alpha = alpha_arg;
		beta = beta_arg;
		xDesc = xDesc_arg;
		xData = xData_arg;
		zDesc = zDesc_arg;
		zData = zData_arg;
		yDesc = yDesc_arg;
		yData = yData_arg;
		bnScaleBiasMeanVarDesc = bnScaleBiasMeanVarDesc_arg;
		bnScaleData = bnScaleData_arg;
		bnBiasData = bnBiasData_arg;
		exponentialAverageFactor = exponentialAverageFactor_arg;
		resultRunningMeanData = resultRunningMeanData_arg;
		resultRunningVarianceData = resultRunningVarianceData_arg;
		epsilon = epsilon_arg;
		saveMean = saveMean_arg;
		saveInvVariance = saveInvVariance_arg;
		activationDesc = activationDesc_arg;
		workspace = workspace_arg;
		workSpaceSizeInBytes = workSpaceSizeInBytes_arg;
		reserveSpace = reserveSpace_arg;
		reserveSpaceSizeInBytes = reserveSpaceSizeInBytes_arg;


	}
	~cudnnBatchNormalizationForwardTrainingEx_record() {}

} cudnnBatchNormalizationForwardTrainingEx_record;


typedef struct  cudnnBatchNormalizationForwardInference_record {

	cudnnHandle_t handle;
	cudnnBatchNormMode_t mode;
	const void *alpha;
	const void *beta;
	cudnnTensorDescriptor_t xDesc;
	const void *x;
	cudnnTensorDescriptor_t yDesc;
	void *y;
	cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
	const void *bnScale;
	const void *bnBias;
	const void *estimatedMean;
	const void *estimatedVariance;
	double epsilon;

	cudnnBatchNormalizationForwardInference_record(cudnnHandle_t handle_arg, cudnnBatchNormMode_t mode_arg, const void *alpha_arg, const void *beta_arg, cudnnTensorDescriptor_t xDesc_arg, const void *x_arg, cudnnTensorDescriptor_t yDesc_arg, void *y_arg, cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc_arg, const void *bnScale_arg, const void *bnBias_arg, const void *estimatedMean_arg, const void *estimatedVariance_arg, double epsilon_arg) {

		handle = handle_arg;
		mode = mode_arg;
		alpha = alpha_arg;
		beta = beta_arg;
		xDesc = xDesc_arg;
		x = x_arg;
		yDesc = yDesc_arg;
		y = y_arg;
		bnScaleBiasMeanVarDesc = bnScaleBiasMeanVarDesc_arg;
		bnScale = bnScale_arg;
		bnBias = bnBias_arg;
		estimatedMean = estimatedMean_arg;
		estimatedVariance = estimatedVariance_arg;
		epsilon = epsilon_arg;

	}

	~cudnnBatchNormalizationForwardInference_record() {}

} cudnnBatchNormalizationForwardInference_record;


typedef struct cudnnRNNForwardInf_record {

	cudnnHandle_t handle;
	cudnnRNNDescriptor_t rnnDesc;
	int seqLength;
	const cudnnTensorDescriptor_t *xDesc;
	const void *x;
    cudnnTensorDescriptor_t hxDesc;
    const void *hx;
    cudnnTensorDescriptor_t cxDesc;
	const void *cx;
	cudnnFilterDescriptor_t wDesc;
	const void *w;
	const cudnnTensorDescriptor_t *yDesc;
	void *y;
	cudnnTensorDescriptor_t hyDesc;
	void *hy;
	cudnnTensorDescriptor_t cyDesc;
	void *cy;
	void *workspace;
	size_t workSpaceSizeInBytes;

	cudnnRNNForwardInf_record(cudnnHandle_t handle_arg, const cudnnRNNDescriptor_t rnnDesc_arg, const int seqLength_arg, const cudnnTensorDescriptor_t *xDesc_arg, const void *x_arg, cudnnTensorDescriptor_t hxDesc_arg, const void *hx_arg, cudnnTensorDescriptor_t cxDesc_arg, const void *cx_arg, cudnnFilterDescriptor_t wDesc_arg, const void *w_arg, const cudnnTensorDescriptor_t *yDesc_arg, void *y_arg, cudnnTensorDescriptor_t hyDesc_arg, void *hy_arg, cudnnTensorDescriptor_t cyDesc_arg, void *cy_arg, void *workspace_arg, size_t workSpaceSizeInBytes_arg) {

		handle = handle_arg;
		rnnDesc = rnnDesc_arg;
		seqLength = seqLength_arg;
		xDesc = xDesc_arg;
		x = x_arg;
		hxDesc = hxDesc_arg;
		hx = hx_arg;
		cxDesc = cxDesc_arg;
		cx = cx_arg;
		wDesc = wDesc_arg;
		w = w_arg;
		yDesc = yDesc_arg;
		y = y_arg;
		hyDesc = hyDesc_arg;
		hy = hy_arg;
		cyDesc = cyDesc_arg;
		cy = cy_arg;
		workspace = workspace_arg;
		workSpaceSizeInBytes = workSpaceSizeInBytes_arg;
	}

	~cudnnRNNForwardInf_record() {}

} cudnnRNNForwardInference_record;


typedef struct cudnnRNNForwardTraining_record {

	cudnnHandle_t handle;
	cudnnRNNDescriptor_t rnnDesc;
	int seqLength;
	const cudnnTensorDescriptor_t *xDesc;
	const void *x;
    cudnnTensorDescriptor_t hxDesc;
    const void *hx;
    cudnnTensorDescriptor_t cxDesc;
    const void *cx;
    cudnnFilterDescriptor_t wDesc;
    const void *w;
    const cudnnTensorDescriptor_t *yDesc;
    void *y;
    cudnnTensorDescriptor_t hyDesc;
    void *hy;
    cudnnTensorDescriptor_t cyDesc;
    void *cy;
    void *workspace;
    size_t workSpaceSizeInBytes;
    void *reserveSpace;
    size_t reserveSpaceSizeInBytes;

	cudnnRNNForwardTraining_record(cudnnHandle_t handle_arg, const cudnnRNNDescriptor_t rnnDesc_arg, const int seqLength_arg, const cudnnTensorDescriptor_t *xDesc_arg, const void *x_arg, const cudnnTensorDescriptor_t hxDesc_arg, const void *hx_arg, const cudnnTensorDescriptor_t cxDesc_arg, const void *cx_arg, const cudnnFilterDescriptor_t wDesc_arg, const void *w_arg, const cudnnTensorDescriptor_t *yDesc_arg, void *y_arg, const cudnnTensorDescriptor_t hyDesc_arg, void *hy_arg, const cudnnTensorDescriptor_t cyDesc_arg, void *cy_arg, void *workspace_arg, size_t workSpaceSizeInBytes_arg, void *reserveSpace_arg, size_t reserveSpaceSizeInBytes_arg) {

			handle = handle_arg;
			rnnDesc = rnnDesc_arg;
			seqLength = seqLength_arg;
			xDesc = xDesc_arg;
			x = x_arg;
			hxDesc = hxDesc_arg;
			hx = hx_arg;
			cxDesc = cxDesc_arg;
			cx = cx_arg;
			wDesc = wDesc_arg;
			w = w_arg;
			yDesc = yDesc_arg;
			y = y_arg;
			hyDesc = hyDesc_arg;
			hy = hy_arg;
			cyDesc = cyDesc_arg;
			cy = cy_arg;
			workspace = workspace_arg;
			workSpaceSizeInBytes = workSpaceSizeInBytes_arg;
			reserveSpace = reserveSpace_arg;
			reserveSpaceSizeInBytes = reserveSpaceSizeInBytes_arg;
	}

	~cudnnRNNForwardTraining_record() {}

} cudnnRNNForwardTraining_record;

typedef struct cudnnCreate_record {
	cudnnHandle_t *handle;
} cudnnCreate_record;

typedef struct cudnnBatchNormReserve_record {

	cudnnHandle_t handle;
	cudnnBatchNormMode_t mode;
	cudnnBatchNormOps_t bnOps;
	cudnnActivationDescriptor_t activationDesc;
	cudnnTensorDescriptor_t xDesc;
    size_t *sizeInBytes;

	cudnnBatchNormReserve_record(cudnnHandle_t handle_arg, cudnnBatchNormMode_t mode_arg, cudnnBatchNormOps_t bnOps_arg, const cudnnActivationDescriptor_t activationDesc_arg, const cudnnTensorDescriptor_t xDesc_arg, size_t *sizeInBytes_arg) {

			handle = handle_arg;
			mode = mode_arg;
			bnOps = bnOps_arg;
			activationDesc = activationDesc_arg;
			xDesc = xDesc_arg;
			sizeInBytes = sizeInBytes_arg;
	}

	~cudnnBatchNormReserve_record() {}

} cudnnBNormReserve_record;


typedef struct cudnnBatchNormalizationBackwardEx_record {

	cudnnHandle_t handle;
    cudnnBatchNormMode_t mode;
    cudnnBatchNormOps_t bnOps;
    const void *alphaDataDiff;
    const void *betaDataDiff;
    const void *alphaParamDiff;
    const void *betaParamDiff;
    cudnnTensorDescriptor_t xDesc;
    const void *xData;
    cudnnTensorDescriptor_t yDesc;
    const void *yData;
	cudnnTensorDescriptor_t dyDesc;
    const void *dyData;
    cudnnTensorDescriptor_t dzDesc;
    void *dzData;
    cudnnTensorDescriptor_t dxDesc;
    void *dxData;
    cudnnTensorDescriptor_t dBnScaleBiasDesc;
    const void *bnScaleData;
    const void *bnBiasData;
    void *dBnScaleData;
    void *dBnBiasData;
    double epsilon;
    const void *savedMean;
    const void *savedInvVariance;
    cudnnActivationDescriptor_t activationDesc;
    void *workspace;
    size_t workSpaceSizeInBytes;
    void *reserveSpace;
    size_t reserveSpaceSizeInBytes;

	cudnnBatchNormalizationBackwardEx_record(
		cudnnHandle_t handle_arg,
		cudnnBatchNormMode_t mode_arg,
		cudnnBatchNormOps_t bnOps_arg,
		const void *alphaDataDiff_arg,
		const void *betaDataDiff_arg,
		const void *alphaParamDiff_arg,
		const void *betaParamDiff_arg,
		const cudnnTensorDescriptor_t xDesc_arg,
		const void *xData_arg,
		const cudnnTensorDescriptor_t yDesc_arg,
		const void *yData_arg,
		const cudnnTensorDescriptor_t dyDesc_arg,
		const void *dyData_arg,
		const cudnnTensorDescriptor_t dzDesc_arg,
		void *dzData_arg,
		const cudnnTensorDescriptor_t dxDesc_arg,
		void *dxData_arg,
		const cudnnTensorDescriptor_t dBnScaleBiasDesc_arg,
		const void *bnScaleData_arg,
		const void *bnBiasData_arg,
		void *dBnScaleData_arg,
		void *dBnBiasData_arg,
		double epsilon_arg,
		const void *savedMean_arg,
		const void *savedInvVariance_arg,
		const cudnnActivationDescriptor_t activationDesc_arg,
		void *workspace_arg,
		size_t workSpaceSizeInBytes_arg,
		void *reserveSpace_arg,
		size_t reserveSpaceSizeInBytes_arg
	) {

		handle = handle_arg;
		mode = mode_arg;
		bnOps = bnOps_arg;
		alphaDataDiff = alphaDataDiff_arg;
		betaDataDiff = betaDataDiff_arg;
		alphaParamDiff = alphaParamDiff_arg;
		betaParamDiff = betaParamDiff_arg;
		xDesc = xDesc_arg;
		xData = xData_arg;
		yDesc = yDesc_arg;
		yData = yData_arg;
		dyDesc = dyDesc_arg;
		dyData = dyData_arg;
		dzDesc = dzDesc_arg;
		dzData = dzData_arg;
		dxDesc = dxDesc_arg;
		dxData = dxData_arg;
		dBnScaleBiasDesc = dBnScaleBiasDesc_arg;
		bnScaleData = bnScaleData_arg;
		bnBiasData = bnBiasData_arg;
		dBnScaleData = dBnScaleData_arg;
		dBnBiasData = dBnBiasData_arg;
		epsilon = epsilon_arg;
		savedMean = savedMean_arg;
		savedInvVariance = savedInvVariance_arg;
		activationDesc = activationDesc_arg;
		workspace = workspace_arg;
		workSpaceSizeInBytes = workSpaceSizeInBytes_arg;
		reserveSpace = reserveSpace_arg;
		reserveSpaceSizeInBytes = reserveSpaceSizeInBytes_arg;
	}

	~cudnnBatchNormalizationBackwardEx_record() {}

} cudnnBatchNormalizationBackwardEx_record;


typedef struct cudnnConvolutionBackwardFilter_record{

	cudnnHandle_t handle;
    const void *alpha;
    cudnnTensorDescriptor_t xDesc;
    const void *x;
    cudnnTensorDescriptor_t dyDesc;
    const void *dy;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionBwdFilterAlgo_t algo;
    void *workSpace;
    size_t workSpaceSizeInBytes;
    const void *beta;
    cudnnFilterDescriptor_t dwDesc;
    void *dw;

	cudnnConvolutionBackwardFilter_record(
		cudnnHandle_t handle_arg,
		const void *alpha_arg,
		const cudnnTensorDescriptor_t xDesc_arg,
		const void *x_arg,
		const cudnnTensorDescriptor_t dyDesc_arg,
		const void *dy_arg,
		const cudnnConvolutionDescriptor_t convDesc_arg,
		cudnnConvolutionBwdFilterAlgo_t algo_arg,
		void *workSpace_arg,
		size_t workSpaceSizeInBytes_arg,
		const void *beta_arg,
		const cudnnFilterDescriptor_t dwDesc_arg,
		void *dw_arg
	)
	{
		handle = handle_arg;
		alpha = alpha_arg;
		xDesc = xDesc_arg;
		x = x_arg;
		dyDesc = dyDesc_arg;
		dy = dy_arg;
		convDesc = convDesc_arg;
		algo = algo_arg;
		workSpace = workSpace_arg;
		workSpaceSizeInBytes = workSpaceSizeInBytes_arg;
		beta = beta_arg;
		dwDesc = dwDesc_arg;
		dw = dw_arg;

	}

	~cudnnConvolutionBackwardFilter_record() {}

 } cudnnConvolutionBackwardFilter_record;


typedef struct cudnnConvolutionBackwardData_record {

	cudnnHandle_t handle;
    const void *alpha;
    cudnnFilterDescriptor_t wDesc;
    const void *w;
    cudnnTensorDescriptor_t dyDesc;
    const void *dy;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionBwdDataAlgo_t algo;
    void *workSpace;
    size_t workSpaceSizeInBytes;
    const void *beta;
    cudnnTensorDescriptor_t dxDesc;
    void *dx;


	cudnnConvolutionBackwardData_record(
		cudnnHandle_t handle_arg,
		const void *alpha_arg,
		const cudnnFilterDescriptor_t wDesc_arg,
		const void *w_arg,
		const cudnnTensorDescriptor_t dyDesc_arg,
		const void *dy_arg,
		const cudnnConvolutionDescriptor_t convDesc_arg,
		cudnnConvolutionBwdDataAlgo_t algo_arg,
		void *workSpace_arg,
		size_t workSpaceSizeInBytes_arg,
		const void *beta_arg,
		const cudnnTensorDescriptor_t dxDesc_arg,
		void *dx_arg
	)
	{
		handle = handle_arg;
		alpha = alpha_arg;
		wDesc = wDesc_arg;
		w = w_arg;
		dyDesc = dyDesc_arg;
		dy = dy_arg;
		convDesc = convDesc_arg;
		algo = algo_arg;
		workSpace = workSpace_arg;
		workSpaceSizeInBytes = workSpaceSizeInBytes_arg;
		beta = beta_arg;
		dxDesc = dxDesc_arg;
		dx = dx_arg;
	}

	~cudnnConvolutionBackwardData_record() {}

} cudnnConvolutionBackwardData_record;

// CUBLAS

typedef struct cublasSgemm_record {

	cublasHandle_t handle;
	cublasOperation_t transa;
	cublasOperation_t transb;
	int m;
	int n;
	int k;
	const float *alpha;
	const float  *A;
       	int lda;
	const float *B;
	int ldb;
	const float *beta;
	float *C;
	int ldc;

	cublasSgemm_record(cublasHandle_t handle_arg, cublasOperation_t transa_arg, cublasOperation_t transb_arg, int m_arg, int n_arg, int k_arg, const float *alpha_arg, const float *A_arg, int lda_arg, const float *B_arg, int ldb_arg, const float *beta_arg, float *C_arg, int ldc_arg) {

		handle = handle_arg;
		transa = transa_arg;
		transb = transb_arg;
		m = m_arg;
		n = n_arg;
		k = k_arg;
		alpha = alpha_arg;
		A = A_arg;
		lda = lda_arg;
		B = B_arg;
		ldb = ldb_arg;
		beta = beta_arg;
		C = C_arg;
		ldc = ldc_arg;
	}

	~cublasSgemm_record() {}

} cublasSgemm_record;


typedef struct cublasSgemmStridedBatched_record {

	cublasHandle_t handle;
	cublasOperation_t transa;
	cublasOperation_t transb;
	int m;
	int n;
	int k;
	const float *alpha;
       	const float *A;
       	int lda;
       	long long int strideA;
       	const float *B;
       	int ldb;
       	long long int strideB;
       	const float *beta;
       	float *C;
       	int ldc;
       	long long int strideC;
       	int batchCount;

	cublasSgemmStridedBatched_record(cublasHandle_t handle_arg, cublasOperation_t transa_arg, cublasOperation_t transb_arg, int m_arg, int n_arg, int k_arg, const float *alpha_arg, const float *A_arg, int lda_arg, long long int strideA_arg, const float *B_arg, int ldb_arg, long long int strideB_arg, const float *beta_arg, float *C_arg, int ldc_arg, long long int strideC_arg, int batchCount_arg) {

		handle = handle_arg;
		transa = transa_arg;
		transb = transb_arg;
		m = m_arg;
		n = n_arg;
		k = k_arg;
		alpha = alpha_arg;
		A = A_arg;
		lda = lda_arg;
		strideA = strideA_arg;
		B = B_arg;
		ldb = ldb_arg;
		strideB = strideB_arg;
		beta = beta_arg;
		C = C_arg;
		ldc = ldc_arg;
		strideC = strideC_arg;
		batchCount = batchCount_arg;
	}


	~cublasSgemmStridedBatched_record() {}


} cublasSgemmStridedBatched_record;

//////////////////////////////////////////////////


enum func_type {
	KERNEL_RECORD,
	MEMCPY_RECORD,
	MALLOC_RECORD,
	FREE_RECORD,
	MEMSET_RECORD,
	CUDNN_CREATE_RECORD,
	CUDNN_CONV_RECORD,
	CUDNN_BNORM_RECORD,
	CUDNN_BNORM_RESERVE_RECORD,
	CUDNN_BNORM_INF_RECORD,
	CUDNN_RNN_INF_RECORD,
	CUDNN_RNN_TRAIN_RECORD,
	CUDNN_CONV_DATA_RECORD,
	CUDNN_CONV_FILTER_RECORD,
	CUDNN_BNORM_BACKWARD_RECORD,
	CUBLAS_SGEMM_RECORD,
	CUBLAS_SGEMM_STRIDED_RECORD
};

union func_data {

	kernel_record krecord;
	cudnnCreate_record cudnnCreateRecord;
	cudnnConvolutionForward_record cudnnConvRecord;
	cudnnBatchNormReserve_record cudnnBNormResRecord;
	cudnnBatchNormalizationForwardTrainingEx_record cudnnBNormRecord;
	cudnnBatchNormalizationForwardInference_record cudnnBNormInfRecord;
	cudnnRNNForwardInf_record cudnnRnnInfRecord;
	cudnnRNNForwardTraining_record cudnnRnnTrainRecord;
	cudnnConvolutionBackwardData_record cudnnConvBackDataRecord;
	cudnnConvolutionBackwardFilter_record cudnnConvBackFilterRecord;
	cudnnBatchNormalizationBackwardEx_record cudnnBNormBackRecord;
	cublasSgemm_record cublasSgemmRecord;
	cublasSgemmStridedBatched_record cublasSgemmStridedRecord;
	memcpy_record mrecord;
	malloc_record malrecord;
	free_record frecord;
	memset_record msetrecord;

	 func_data() {}
	 ~func_data() {};
};

typedef struct func_record {

	enum func_type type;
	union func_data data;

} func_record;


// globals
extern volatile pid_t* thread_ids; // 2*N threads + scheduler

extern queue<func_record>** kqueues;
extern pthread_mutex_t** mutexes;

extern int* func_indexes;
extern int* num_total_clients;

extern volatile bool** client_request_status;
extern volatile bool* client_stop;
extern volatile bool* client_stop_ack;
extern volatile bool* affinity_set;

// functions
extern cudaError_t (*kernel_func)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
extern cudaError_t (*memcpy_func)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
extern cudaError_t (*memcpy_async_func)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern cudaError_t (*malloc_func)(void** devPtr, size_t size);
extern cudaError_t (*free_func)(void* devPtr);
extern cudaError_t (*memset_func)(void* devPtr, int  value, size_t count);
extern cudaError_t (*memset_async_func)(void* devPtr, int  value, size_t count, cudaStream_t stream);

extern cudnnStatus_t (*cudnn_conv_func)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) ;
extern cudnnStatus_t (*cudnn_bnorm_func)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc,  const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScaleData, const void *bnBiasData, double exponentialAverageFactor, void *resultRunningMeanData, void *resultRunningVarianceData, double epsilon, void *saveMean, void *saveInvVariance, const cudnnActivationDescriptor_t activationDesc,  void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes);
extern cudnnStatus_t (*cudnn_bnorm_infer_func)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon);
extern cudnnStatus_t (*cudnn_rnn_func)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes);
extern cudnnStatus_t (*cudnn_rnn_train_func)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes);
extern cublasStatus_t (*cublas_sgemm_func)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
extern cublasStatus_t (*cublas_sgemm_strided_func)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount);

extern cudnnStatus_t (*cudnn_bnorm_bw_func)(
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
);

extern cudnnStatus_t (*cudnn_conv_bw_data_func)(
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
);

extern cudnnStatus_t (*cudnn_conv_bw_filter_func)(
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
);


// util functions
int get_idx();
void block(int idx, pthread_mutex_t** mutexes, queue<func_record>** kqueues);


#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
		           const int line) {
	if (err != cudaSuccess)
	{
		printf("CUDA Runtime Error at: %s:%d\n", file, line);
		printf("Error %d, %s\n", err, cudaGetErrorString(err));
	}
	assert (err == cudaSuccess);
}
