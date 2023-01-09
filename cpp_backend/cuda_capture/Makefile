NVCC=/usr/local/cuda-10.2/bin/nvcc

libint.so: intercept_custom.cu
	$(NVCC) -Xcompiler -fPIC -G -g -o libint.so --shared intercept_custom.cu -lcudart

libinttemp.so: intercept_temp.cu
	$(NVCC) -Xcompiler -fPIC -G -g -o libinttemp.so --shared intercept_temp.cu -lcudart

all:
	make libint.so libinttemp.so

clean:
	rm -rf libint.so
