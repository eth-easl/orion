NVCC=/usr/local/cuda-10.2/bin/nvcc

libinttemp.so: intercept_temp.cu
	$(NVCC) -Xcompiler -fPIC -G -DDEBUG -g -o libinttemp.so --shared intercept_temp.cu -lcudart -lcudnn

all:
	make libinttemp.so

clean:
	rm -rf libint.so
