#include "utils.h"

void register_functions() {

    // for kernel
	*(void **)(&kernel_function) = dlsym(RTLD_DEFAULT, "cudaLaunchKernel");

	// for memcpy
	*(void **)(&memcpy_function) = dlsym (RTLD_DEFAULT, "cudaMemcpy");

	// for memcpy_async
	*(void **)(&memcpy_async_function) = dlsym (RTLD_DEFAULT, "cudaMemcpyAsync");

	// for malloc
	*(void **)(&malloc_function) = dlsym (RTLD_DEFAULT, "cudaMalloc");

	// for cudnn conv
	*(void **)(&cudnn_conv_function) = dlsym(RTLD_DEFAULT, "cudnnConvolutionForward");
	assert(cudnn_conv_function != NULL);

	// for bnorm train
	*(void **)(&cudnn_bnorm_function) = dlsym(RTLD_DEFAULT, "cudnnBatchNormalizationForwardTrainingEx");
	assert(cudnn_bnorm_function != NULL);

	// for bnorm infer
	*(void **)(&cudnn_bnorm_infer_function) = dlsym(RTLD_DEFAULT, "cudnnBatchNormalizationForwardInference");
	assert(cudnn_bnorm_infer_function != NULL);

	// for rnn infer
	*(void **)(&cudnn_rnn_function) = dlsym(RTLD_DEFAULT, "cudnnRNNForwardInference");
	assert(cudnn_rnn_function != NULL);

	// CUBLAS sgemm
	*(void **)(&cublas_sgemm_function) = dlsym(RTLD_DEFAULT, "cublasSgemm_v2");
	assert(cublas_sgemm_function != NULL);


	// CUBLAS sgemm strided
	*(void **)(&cublas_sgemm_strided_function) = dlsym(RTLD_DEFAULT, "cublasSgemmStridedBatched");
	assert(&cublas_sgemm_strided_function != NULL);

}


void schedule_kernel(struct func_record frecord, cudaStream_t sched_stream) {

    // case 1
	if (frecord.type == KERNEL_RECORD) {
		//DEBUG_PRINT("found a new kernel record! kernel func is %p\n", kernel_function);
		kernel_record record = frecord.data.krecord;
		(*kernel_function)(record.func, record.gridDim, record.blockDim, record.args, record.sharedMem, sched_stream);
	}

	else if (frecord.type == MEMCPY_RECORD) {
		//DEBUG_PRINT("found a new memcpy record!\n");
		memcpy_record record = frecord.data.mrecord;
		if (not record.async) {
			(*memcpy_function)(record.dst, record.src, record.count, record.kind);
		} else {
			(*memcpy_async_function)(record.dst, record.src, record.count, record.kind, sched_stream);
		}

	}

	else if (frecord.type == MALLOC_RECORD) {
		//DEBUG_PRINT("found a new malloc record!\n");
		malloc_record record = frecord.data.malrecord;
		(*malloc_function)(record.devPtr, record.size);
    }

	else if (frecord.type == CUDNN_CONV_RECORD) {
	    //DEBUG_PRINT("found a new cudnn conv record!\n");
		cudnnConvolutionForward_record record = frecord.data.cudnnConvRecord;
		cudnnSetStream(record.handle, 0);
		(*cudnn_conv_function)(record.handle, record.alpha, record.xDesc, record.x, record.wDesc, record.w, record.convDesc, record.algo, record.workSpace, record.workSpaceSizeInBytes, record.beta, record.yDesc, record.y);
		cudnnSetStream(record.handle, 0); // TODO: I want to set the default stream here
	}

	else if (frecord.type == CUDNN_BNORM_RECORD) {
		//DEBUG_PRINT("found a new bnorm record!\n");
		cudnnBatchNormalizationForwardTrainingEx_record record = frecord.data.cudnnBNormRecord;
		cudnnSetStream(record.handle, 0);
		(*cudnn_bnorm_function)(record.handle, record.mode, record.bnOps, record.alpha, record.beta, record.xDesc, record.xData, record.zDesc, record.zData, record.yDesc, record.yData, record.bnScaleBiasMeanVarDesc, record.bnScaleData, record.bnBiasData, record.exponentialAverageFactor, record.resultRunningMeanData, record.resultRunningVarianceData, record.epsilon, record.saveMean, record.saveInvVariance, record.activationDesc, record.workspace, record.workSpaceSizeInBytes, record.reserveSpace, record.reserveSpaceSizeInBytes);
		cudnnSetStream(record.handle, 0); // TODO: I want to set the default stream here
	}

	else if (frecord.type == CUDNN_BNORM_INF_RECORD) {
		//DEBUG_PRINT("found a new bnorm inf record!\n");
		cudnnBatchNormalizationForwardInference_record record = frecord.data.cudnnBNormInfRecord;
		cudnnSetStream(record.handle, 0);
		(*cudnn_bnorm_infer_function)(record.handle, record.mode, record.alpha, record.beta, record.xDesc, record.x, record.yDesc, record.y, record.bnScaleBiasMeanVarDesc, record.bnScale, record.bnBias, record.estimatedMean, record.estimatedVariance, record.epsilon);
		cudnnSetStream(record.handle, 0);
	}

	else if (frecord.type == CUDNN_RNN_INF_RECORD) {
		DEBUG_PRINT("found a new cudnn rnn inf record!\n");
		cudnnRNNForwardInference_record record = frecord.data.cudnnRnnInfRecord;
		(*cudnn_rnn_function)(record.handle, record.rnnDesc, record.seqLength, record.xDesc, record.x, record.hxDesc, record.hx, record.cxDesc, record.cx, record.wDesc, record.w, record.yDesc, record.y, record.hyDesc, record.hy, record.cyDesc, record.cy, record.workspace, record.workSpaceSizeInBytes);
	}

    else if (frecord.type == CUBLAS_SGEMM_RECORD) {
		//DEBUG_PRINT("found a new sgemm record!\n");

		// TODO: what to do about streams?
		cublasSgemm_record record = frecord.data.cublasSgemmRecord;
		DEBUG_PRINT("handle is %p\n", record.handle);
		(*cublas_sgemm_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.B, record.ldb, record.beta, record.C, record.ldc);
	}

	else if (frecord.type == CUBLAS_SGEMM_STRIDED_RECORD) {
		DEBUG_PRINT("found a new sgemm strided record!\n");

		cublasSgemmStridedBatched_record record = frecord.data.cublasSgemmStridedRecord;
		DEBUG_PRINT("handle is %p\n", record.handle);
		(*cublas_sgemm_strided_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.strideA, record.B, record.ldb, record.strideB, record.beta, record.C, record.ldc, record.strideC, record.batchCount);
    }

}