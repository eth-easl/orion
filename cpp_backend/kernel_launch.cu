#include "kernel_launch.h"

using namespace std;

// just a toy example
__global__ void toy_kernel(int num) {
	printf("------- Num is %d\n", num);
}


__global__ void toy_kernel_two(int num1, float num2) {
	printf("--------- Num1 is %d, num2 is %f\n", num1, num2);
}


void Operator::execute() {
	cudaLaunchKernel((void*)func, dim3(1), dim3(1), args, 0, NULL);
}



__global__ void toy_kernel_ar(int* num, int val) {
	
	printf("num is: %p\n", num);
	for (int i=0; i<10; i++) {
		num[i] = val;
		printf("------- Num is %d\n", num[i]);
	}
}


