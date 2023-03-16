NVCC=/usr/local/cuda-10.2/bin/nvcc

libinttemp.so: utils.cu intercept_cudnn.cu intercept_cublas.cu intercept_temp.cu
	$(NVCC) -Xcompiler -fPIC -G -DDEBUG -DINCLUDE_TORCH_CUDA -g  -o libinttemp.so --shared --extended-lambda  utils.cu intercept_cudnn.cu intercept_cublas.cu intercept_temp.cu -I $(HOME)/pytorch/aten/src/ATen/native/cuda -I $(HOME)/pytorch/ -I $(HOME)/pytorch/aten/src/  -L $(HOME)/pytorch/build/lib -lshm -lcudart -lcudnn -lcublas

libintsync.so: intercept_sync.cu
	$(NVCC) -Xcompiler -fPIC -G -DDEBUG -DINCLUDE_TORCH_CUDA -g  -o libintsync.so --shared intercept_sync.cu -lcudart -lcudnn

all:
	make libinttemp.so

clean:
	rm -rf *.o libinttemp.so libintsync.so
