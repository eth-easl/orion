CC=g++
NVCC=/usr/local/cuda-10.2/bin/nvcc
CFLAGS=-O0
CUDAINCLUDE=/usr/local/cuda-10.2/include/
CUDALIB=/usr/local/cuda-10.2/lib64

kernel_launch.o: kernel_launch.cu
	$(NVCC) $(CFLAGS) -Xcompiler -fPIC -c kernel_launch.cu

scheduler.o: scheduler.cpp
	$(CC) $(CFLAGS) -fPIC -shared -c scheduler.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lpthread

scheduler.so: scheduler.o kernel_launch.o
	 $(CC) $(CFLAGS) -fPIC -shared kernel_launch.o scheduler.o -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lpthread -o scheduler.so


example.o: toy_example.cpp
	$(CC) $(CFLAGS) -c toy_example.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lpthread

example: kernel_launch.o scheduler.o example.o
	$(CC) -o example scheduler.o kernel_launch.o toy_example.o -L$(CUDALIB) -lcudart -ldl -lpthread


all:
	make example

clean:
	rm -rf *.o example scheduler.so
