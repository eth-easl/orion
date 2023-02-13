NVCC=/usr/local/cuda-10.2/bin/nvcc

libinttemp.so: intercept_temp.cu
	$(NVCC) -Xcompiler -fPIC -G -DDEBUG -g  -o libinttemp.so --shared intercept_temp.cu -I /home/image-varuna/pytorch/aten/src/ATen/native/cuda -I /home/image-varuna/pytorch/ -L /home/image-varuna/pytorch/build/lib -lshm -lcudart -lcudnn

libintsync.so: intercept_sync.cu
	$(NVCC) -Xcompiler -fPIC -G -DDEBUG -g  -o libintsync.so --shared intercept_sync.cu -lcudart -lcudnn

all:
	make libinttemp.so libintsync.so

clean:
	rm -rf libinttemp.so libintsync.so
