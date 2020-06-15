#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>

constexpr size_t threadsPerBlock = 256;
constexpr size_t blocks = 1024;

__global__ void sieve_small(unsigned long long * const table, const size_t size,
    const unsigned int * const primes, const size_t prime_num,
    const unsigned long long * const mask, const size_t mask_pitch,
    const uint64_t offset) {
  size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  int rem[18];
  for (int i = 0; i < prime_num; ++i) {
    rem[i] = (offset + index * 64) % primes[i];
  }
  while (index < size) {
    unsigned long long bits = ~0ULL;
    for (int i = 0; i < prime_num; ++i) {
      bits &= mask[i*mask_pitch + rem[i]];
    }
    table[index] = bits;
    index += blockDim.x * gridDim.x;
    for (int i = 0; i < prime_num; ++i) {
      rem[i] += blockDim.x * gridDim.x * 64;
      rem[i] %= primes[i];
    }
  }
}

constexpr size_t small_table_size = 8192;
__global__ void sieve_middle(unsigned int * const table, const size_t size,
    const unsigned int * const primes, const size_t prime_num,
    const uint64_t offset) {
  __shared__ unsigned int small_table[small_table_size];
  for (int table_index = blockIdx.x; table_index * small_table_size < size; table_index += gridDim.x) {
    for (int i = threadIdx.x; i < small_table_size; i += blockDim.x) {
      small_table[i] = ~0;
    }
    __syncthreads();
    const uint64_t offset_small = offset + table_index * small_table_size * 32;
    int index = threadIdx.x;
    while (index < prime_num) {
      const unsigned int prime = primes[index];
      unsigned int i = (prime - (offset_small % prime)) % prime;
      while (i < small_table_size * 32) {
        unsigned int word_index = i / 32;
        unsigned int bit_index = i % 32;
        atomicAnd(small_table + word_index, ~(1 << bit_index));
        i += prime;
      }
      index += blockDim.x;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < small_table_size; i += blockDim.x) {
      table[table_index * small_table_size + i] &= small_table[i];
    }
    __syncthreads();
  }
}

__global__ void sieve(unsigned int * const table, const size_t width,
    const unsigned int * const primes, const size_t prime_num,
    const uint64_t offset) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  while (index < prime_num) {
    const unsigned int prime = primes[index];
    unsigned int i = (prime - (offset % prime)) % prime;
    while (i < width) {
      unsigned int word_index = i / 32;
      unsigned int bit_index = i % 32;
      atomicAnd(table + word_index, ~(1 << bit_index));
      i += prime;
    }
    index += blockDim.x * gridDim.x;
  }
}

__global__ void count(const unsigned long long * const table, const size_t size,
    unsigned int * sum) {
  __shared__ unsigned int cache[threadsPerBlock];
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int cacheIndex = threadIdx.x;
  cache[cacheIndex] = 0;
  while (index < size) {
    cache[cacheIndex] += __popcll(table[index]);
    index += blockDim.x * gridDim.x;
  }
  __syncthreads();
  int i = threadsPerBlock / 2;
  while (i) {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();
    i /= 2;
  }
  if (cacheIndex == 0)
    sum[blockIdx.x] = cache[0];
}

class bit_array {
  std::vector<uint64_t> data;
 public:
  bit_array(const size_t size, bool init)
    : data((size + 63) / 64, init ? ~UINT64_C(0) : 0) {}
  bit_array(const size_t size, uint64_t init)
    : data((size + 63) / 64, init) {}
  void set(const size_t index) {
    data[index / 64] |= UINT64_C(1) << (index % 64);
  }
  void reset(const size_t index) {
    data[index / 64] &= ~(UINT64_C(1) << (index % 64));
  }
  bool test(const size_t index) const {
    return 1 & (data[index / 64] >> (index % 64));
  }
};

thrust::host_vector<unsigned int> primes_list(const uint64_t sqrtn) {
  const uint64_t n = sqrtn * sqrtn;
  bit_array is_prime(n+1, UINT64_C(0xAAAAAAAAAAAAAAAA));
  for (int i = 3; i <= sqrtn; i += 2)
    if (is_prime.test(i))
      for (int j = i*i; j <= n; j += 2*i)
        is_prime.reset(j);
  thrust::host_vector<unsigned int> primes;
  primes.push_back(2);
  for (unsigned int i = 3; i <= n; i += 2)
    if (is_prime.test(i))
      primes.push_back(i);
  return primes;
}

void generate_mask(const thrust::host_vector<unsigned int> &primes,
    const size_t primes_under_64,
    unsigned long long * const mask_host, const size_t pitch_in_words) {
  for (int i = 0; i < primes_under_64; ++i) {
    unsigned long long mask_rev = (1ULL << primes[i]) + 1;
    for (int shift = primes[i] * 2; shift < 64; shift *= 2) {
      mask_rev |= mask_rev << shift;
    }
    const int shift = ((64 + primes[i] - 1) / primes[i]) * primes[i];
    for (int j = 0; j < primes[i]; ++j) {
      if (shift - j >= 64) {
        mask_host[i * pitch_in_words + j] = ~(mask_rev >> j);
      } else {
        mask_host[i * pitch_in_words + j] = ~((mask_rev >> j) | (mask_rev << (shift - j)));
      }
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << argv[0] << " LOOP_COUNT" << std::endl;
    exit(EXIT_FAILURE);
  }
  const uint64_t sqrtn = 4096;
  const uint64_t n = sqrtn * sqrtn;
  const uint64_t loop_count = std::atoi(argv[1]);
  const uint64_t N = n * loop_count;
  thrust::host_vector<unsigned int> primes = primes_list(sqrtn);
  std::cout << primes.size() << std::endl;
  const size_t primes_under_64 = thrust::upper_bound(primes.begin(), primes.end(), 64) - primes.begin();
  const size_t primes_under_middle = thrust::upper_bound(primes.begin(), primes.end(), 131072) - primes.begin();
  unsigned long long *mask_dev = nullptr;
  size_t pitch = 0;
  cudaMallocPitch(reinterpret_cast<void**>(&mask_dev), &pitch, sizeof(unsigned long long) * 64, primes_under_64);
  unsigned long long *mask_host = (unsigned long long *)malloc(primes_under_64 * pitch);
  size_t pitch_in_words = pitch / sizeof(unsigned long long);
  generate_mask(primes, primes_under_64, mask_host, pitch_in_words);
  cudaMemcpy(mask_dev, mask_host, primes_under_64 * pitch, cudaMemcpyHostToDevice);
  thrust::device_vector<unsigned int> primes_dev = primes;
  thrust::device_vector<unsigned long long> table_dev(n/64);
  thrust::device_vector<unsigned int> sum_dev(blocks);
  thrust::host_vector<unsigned int> sum;
  for (uint64_t offset = n; offset < N; offset += n) {
    const size_t primes_under_sqrt_max = thrust::upper_bound(primes.begin(), primes.end(), sqrt(offset + n)) - primes.begin();
    sieve_small<<<blocks, threadsPerBlock>>>(table_dev.data().get(), table_dev.size(),
        primes_dev.data().get(), primes_under_64, mask_dev, pitch_in_words, offset);
    sieve_middle<<<blocks, threadsPerBlock>>>((unsigned int *)table_dev.data().get(), table_dev.size() * 2,
        primes_dev.data().get()+primes_under_64, std::min(primes_under_middle, primes_under_sqrt_max) - primes_under_64, offset);
    if (primes_under_sqrt_max > primes_under_middle) {
      sieve<<<blocks, threadsPerBlock>>>((unsigned int *)table_dev.data().get(), n,
          primes_dev.data().get()+primes_under_middle, primes_under_sqrt_max - primes_under_middle, offset);
    }
    count<<<blocks, threadsPerBlock>>>(table_dev.data().get(), table_dev.size(), sum_dev.data().get());
    sum = sum_dev;
    uint64_t sum_all = 0;
    for (const unsigned int sumb : sum) sum_all += sumb;
    std::cout << sum_all << '\n';
  }
  cudaFree(mask_dev);
  free(mask_host);
  return 0;
}
