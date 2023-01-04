NVCC=/usr/local/cuda-11.4/bin/nvcc

libint.so: intercept_custom.cu
	$(NVCC) -Xcompiler -fPIC -o libint.so --shared intercept_custom.cu -lcudart

all:
	make libint.so

clean:
	rm -rf libint.so
