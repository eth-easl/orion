#include <cuda_runtime.h>
#include <cuda.h>

#include "../system_utils.h"
#include "../cuda_capture/intercept_temp.h"

extern cudaError_t (*kernel_function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
extern cudaError_t (*memcpy_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
extern cudaError_t (*memcpy_async_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern cudaError_t (*malloc_function)(void** devPtr, size_t size);
extern cudaError_t (*free_function)(void* devPtr);
extern cudaError_t (*memset_function)(void* devPtr, int  value, size_t count);
extern cudaError_t (*memset_async_function)(void* devPtr, int  value, size_t count, cudaStream_t stream);

extern cudnnStatus_t (*cudnn_create_function)(cudnnHandle_t *handle);
extern cudnnStatus_t (*cudnn_bnorm_reserve_function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes);
extern cudnnStatus_t (*cudnn_conv_function)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) ;
extern cudnnStatus_t (*cudnn_bnorm_function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc,  const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScaleData, const void *bnBiasData, double exponentialAverageFactor, void *resultRunningMeanData, void *resultRunningVarianceData, double epsilon, void *saveMean, void *saveInvVariance, const cudnnActivationDescriptor_t activationDesc,  void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes);
extern cudnnStatus_t (*cudnn_bnorm_infer_function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon);
extern cudnnStatus_t (*cudnn_rnn_function)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes);
extern cudnnStatus_t (*cudnn_rnn_train_function)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes);
extern cublasStatus_t (*cublas_sgemm_function)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
extern cublasStatus_t (*cublas_sgemm_strided_function)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount);

extern cudnnStatus_t (*cudnn_bnorm_bw_function)(
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

extern cudnnStatus_t (*cudnn_conv_bw_data_function)(
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

extern cudnnStatus_t (*cudnn_conv_bw_filter_function)(
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

typedef struct op_info {

	string name;
	int profile; // 1: compute-bound, 0: mem-bound, -1: unclear
	int mem;
	int sm_used;
	float duration;

} op_info;

void register_functions();
void schedule_kernel(struct func_record frecord, cudaStream_t* sched_stream, int idx, cudaEvent_t* event, int* seen, int* event_ids, int evid);
void schedule_pair(
	vector<func_record*> &frecords,
	queue<struct func_record>** &buffers,
	pthread_mutex_t** &mutexes,
	vector<vector<op_info>> &op_info_vector,
	int* seen, int max_sms,
	cudaStream_t** sched_streams,
	int* streams,
	cudaEvent_t*** events,
	int num_events,
	int* event_ids
);
void schedule_pair_kernel_padding(
	vector<func_record*> &frecords,
	queue<struct func_record>** &cbuffers,
	pthread_mutex_t** &cmutexes,
	vector<vector<op_info>> &op_info_vector,
	int* seen, int max_sms,
	cudaStream_t** sched_streams,
	int* streams,
	cudaEvent_t*** events,
	int num_events,
	int* event_ids
);
void pop_from_queue(queue<struct func_record>* client_queue, pthread_mutex_t* client_mutex, int idx);
void create_streams(cudaStream_t** sched_streams, int num, bool reef);
void create_events(cudaEvent_t*** events, int num);
void wait_for_stream(int idx, int profile, int current_prio, int prev_prio, cudaStream_t* sched_stream, cudaEvent_t*** events, int num_events, int* event_ids);
void wait_all_streams(int idx, cudaStream_t* sched_stream, cudaEvent_t*** events, int num_events, int* event_ids);
void process_eval(vector<vector<float>> &client_durations);