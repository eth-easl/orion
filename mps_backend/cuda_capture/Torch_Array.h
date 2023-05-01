#pragma once

// A fixed-size array type usable from both host and
// device code.


template <typename T, int size_>
struct Array {
  
  T data[size_];

  __host__ __device__ T operator[](int i) const {
    return data[i];
  }
  __host__ __device__ T& operator[](int i) {
    return data[i];
  }
#if defined(USE_ROCM)
  __host__ __device__ Array() = default;
  __host__ __device__ Array(const Array&) = default;
  __host__ __device__ Array& operator=(const Array&) = default;
#else
  Array() = default;
  Array(const Array&) = default;
  Array& operator=(const Array&) = default;
#endif
  static constexpr int size(){return size_;}
  // Fill the array with x.
  __host__ __device__ Array(T x) {
    for (int i = 0; i < size_; i++) {
      data[i] = x;
    }
  }
};

