#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>
//#include <cuda_runtime.h>
//#include <driver_types.h>

void cudaLaunchKernelHelper (CUstream hStream);

CUresult cudaLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) {
	
	printf("hello!\n");

	void* handle;

	CUresult (*function)(CUfunction f,  
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, 
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra);


	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaLaunchKernel");

	cudaLaunchKernelHelper (hStream);

	(*function)(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);

}

void cudaLaunchKernelHelper (CUstream hStream) {
	// Nothing
	printf ("cudaLaunchHelper\n");
}
