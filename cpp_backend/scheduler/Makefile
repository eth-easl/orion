CC=g++
NVCC=/usr/local/cuda-10.2/bin/nvcc
CFLAGS=-O3
CUDAINCLUDE=/usr/local/cuda-10.2/include/
CUDALIB=/usr/local/cuda-10.2/lib64

utils_sched.o: utils_sched.cpp
	$(NVCC) $(CFLAGS) -Xcompiler  -fPIC -x cu -shared -c utils_sched.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lpthread

sleep.o: sleep.cu
	$(NVCC) $(CFLAGS) -Xcompiler -fPIC -x cu -c sleep.cu --shared -lcudart

scheduler.o: scheduler.cpp
	$(NVCC) $(CFLAGS) -Xcompiler  -fPIC  -x cu -shared -c scheduler.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lpthread

scheduler.so: scheduler.o utils_sched.o sleep.o
	 $(CC) $(CFLAGS) -fPIC -shared utils_sched.o sleep.o scheduler.o -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lpthread -o scheduler.so


all:
	make scheduler.so

clean:
	rm -rf *.o scheduler.so
