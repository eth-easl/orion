CC=g++
NVCC=/usr/local/cuda-10.2/bin/nvcc
CFLAGS=-O3 -DDEBUG
CUDAINCLUDE=/usr/local/cuda-10.2/include/
CUDALIB=/usr/local/cuda-10.2/lib64

utils.o: utils.cpp
	$(CC) $(CFLAGS) -fPIC -shared -c utils.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lpthread

scheduler.o: scheduler.cpp
	$(CC) $(CFLAGS) -fPIC -shared -c scheduler.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lpthread

scheduler.so: scheduler.o kernel_launch.o
	 $(CC) $(CFLAGS) -fPIC -shared kernel_launch.o scheduler.o -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lpthread -o scheduler.so


all:
	make scheduler.so

clean:
	rm -rf *.o scheduler.so
