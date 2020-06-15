sieve: sieve.cu
	nvcc -o sieve -std=c++11 -O2 -arch=sm_61 sieve.cu
