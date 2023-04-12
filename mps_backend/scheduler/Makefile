CC=g++
NVCC=/usr/local/cuda-10.2/bin/nvcc
CFLAGS=-O3
CUDAINCLUDE=/usr/local/cuda-10.2/include/
CUDALIB=/usr/local/cuda-10.2/lib64

scheduler.o: scheduler.cpp
	$(NVCC) $(CFLAGS) -Xcompiler  -fPIC  -x cu -shared -c scheduler.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lpthread

scheduler.so: scheduler.o
	 $(CC) $(CFLAGS) -fPIC -shared scheduler.o -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lpthread -o scheduler.so

all:
	make scheduler.so

clean:
	rm -rf *.o *.so
