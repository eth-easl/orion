#include <stdio.h>

__global__ void toy_kernel(int num);
__global__ void toy_kernel_two(int num1, float num2);
__global__ void toy_kernel_ar(int* num, int val); 

class Operator {
	
	public:
		void* func;
   		void** args;

		Operator(): func(NULL), args(NULL) {}
		Operator(void* f): func(f), args(NULL) {}
		Operator(void* f, void** cargs): func(f), args(cargs) {}				
		void execute();
};
