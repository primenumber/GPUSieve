sieve: sieve.cu
	nvcc -o sieve -std=c++14 -O2 -arch=sm_61 sieve.cu
