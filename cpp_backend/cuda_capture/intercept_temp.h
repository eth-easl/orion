#include <dlfcn.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <queue>
#include <string>
#include <pthread.h>
#include <cuda.h>
#include <cudnn.h>
#include <cublas.h>
#include <sys/types.h>
#include <syscall.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <assert.h>
#include <vector>
#include "Torch_Array.h"

#ifdef INCLUDE_TORCH_CUDA
#include "Reduce.cuh"
#include "ATen/native/SharedReduceOps.h"
#include "ATen/NumericUtils.h"
#endif

#define VECTORIZED_ELEMENTWISE_KERNEL "void at::native::vectorized_elementwise_kernel"
#define CUB_DEVICE_REDUCE_SINGLE_TILE_KERNEL "void at_cuda_detail::cub::DeviceReduceSingleTileKernel"
#define CUB_DEVICE_COMPACT_INIT_KERNEL "void at_cuda_detail::cub::DeviceCompactInitKernel"
#define CUB_DEVICE_SELECT_SWEEP_KERNEL "void at_cuda_detail::cub::DeviceSelectSweepKernel"
#define INDEX_ELEMENTWISE_KERNEL "void at::native::index_elementwise_kernel"
#define UNROLLED_ELEMENTWISE_KERNEL "void at::native::unrolled_elementwise_kernel"
#define REDUCE_KERNEL "void at::native::reduce_kernel"
#define MAX_POOL_FORWARD_NCHW "void at::native::(anonymous namespace)::max_pool_forward_nchw"

#define MOBILENET "mobilenet"
#define VGG16 "vgg16"
#define RESNET50 "resnet50"
#define RESNET101 "resnet101"

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

//////////////////////////////////////////////////


enum func_type {KERNEL_RECORD, MEMCPY_RECORD, MALLOC_RECORD, FREE_RECORD, CUDNN_CONV_RECORD, CUDNN_BNORM_RECORD, CUDNN_BNORM_INF_RECORD, CUBLAS_SGEMM_RECORD};

union func_data {

	kernel_record krecord;
	cudnnConvolutionForward_record cudnnConvRecord;
	cudnnBatchNormalizationForwardTrainingEx_record cudnnBNormRecord;
	cudnnBatchNormalizationForwardInference_record cudnnBNormInfRecord;
	cublasSgemm_record cublasSgemmRecord;
	memcpy_record mrecord;
	malloc_record malrecord;
	free_record frecord;

	 func_data() {}
	 ~func_data() {};
};

typedef struct func_record {
	
	enum func_type type;
	union func_data data;

} func_record;

