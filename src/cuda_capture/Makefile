NVCC=g++ #/usr/local/cuda-11.0/bin/nvcc

libinttemp.so: utils_interc.cpp intercept_cudnn.cpp intercept_cublas.cpp intercept_temp.cpp
	$(NVCC) -fPIC -O3 -L/usr/local/cuda-10.2/lib64 -I/usr/local/cuda-10.2/include  -g  -o libinttemp.so --shared utils_interc.cpp intercept_cudnn.cpp intercept_cublas.cpp intercept_temp.cpp -lcudart

all:
	make libinttemp.so

clean:
	rm -rf *.o libinttemp.so
