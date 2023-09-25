NVCC=/usr/local/cuda-10.2/bin/nvcc

libinttemp.so: utils_interc.cu intercept_cudnn.cu intercept_cublas.cu intercept_temp.cu
	$(NVCC) -Xcompiler -fPIC -O3 -g  -o libinttemp.so --shared --extended-lambda  utils_interc.cu intercept_cudnn.cu intercept_cublas.cu intercept_temp.cu -I $(HOME)/pytorch/aten/src/ATen/native/cuda -I $(HOME)/pytorch/ -I $(HOME)/pytorch/aten/src/  -L $(HOME)/pytorch/build/lib -lshm -lcudart -lcudnn -lcublas

all:
	make libinttemp.so

clean:
	rm -rf *.o libinttemp.so
