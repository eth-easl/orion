import pandas as pd
import sys

pwd = sys.argv[1]
df = pd.read_csv(f'{pwd}/output_ncu.csv')
kernels = []
metrics_to_get = ['Duration', 'Block Size', 'Grid Size', 'Compute (SM) [%]', 'DRAM Throughput', 'Registers Per Thread', 'Static Shared Memory Per Block']

unique_kernel_names = set()

for index, row in df.iterrows():
    kernel = row['Kernel Name']
    metric_name = row['Metric Name']

    if metric_name == 'DRAM Frequency':
        kernels.append([kernel])
        unique_kernel_names.add(kernel)
    elif metric_name in metrics_to_get:
        kernels[-1].append(row['Metric Value'])

for x in unique_kernel_names:
    print(x)
    print("------------------------------------")


for kernel in kernels:
    num_threads = int(kernel[-2]) * int(kernel[-3])
    num_registers = num_threads * int(kernel[-1])
    kernel += [num_threads, num_registers]

'''
# mem_region in roofline, and type of layer
kernels_additional_info = {}

# BERT
kernels_additional_info['void at::native::<unnamed>::indexSelectLargeIndex<float, long, unsigned int, (int)2, (int)2, (int)-2, (bool)1>(at::cuda::detail::TensorInfo<T1, T3>, at::cuda::detail::TensorInfo<T1, T3>, at::cuda::detail::TensorInfo<T2, T3>, int, int, T3, T3, long)'] = [-1, 'Embedding']
kernels_additional_info['volta_sgemm_128x32_tn'] = [0, 'Linear']
kernels_additional_info['volta_sgemm_128x64_nn'] = [0, 'MatMul']
kernels_additional_info['volta_sgemm_128x64_tn'] = [0, 'Linear']
kernels_additional_info['void <unnamed>::softmax_warp_forward<float, float, float, (int)7, (bool)0>(T2 *, const T1 *, int, int, int)'] = [1, 'Softmax']
kernels_additional_info['void splitKreduce_kernel<float, float, float>(cublasSplitKParams<T3>, const T1 *, const T2 *, T2 *, const T3 *, const T3 *)'] = [1, 'Linear']
kernels_additional_info['volta_sgemm_32x128_tn'] = [0, 'Linear']
kernels_additional_info['void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::GeluCUDAKernelImpl(at::TensorIteratorBase &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 4)]::operator ()() const::[lambda(float) (instance 1)], at::detail::Array<char *, (int)2>>(int, T2, T3)'] = [1, 'Gelu']
kernels_additional_info['void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float>>, at::detail::Array<char *, (int)3>>(int, T2, T3)'] = [1, 'Add']
kernels_additional_info['void cuApplyLayerNorm<float, float, float>(T3 *, T2 *, T2 *, const T1 *, int, int, T2, const T3 *, const T3 *)'] = [1, 'Norm']
kernels_additional_info['volta_sgemm_32x32_sliced1x4_tn'] = [0, 'Linear']
kernels_additional_info['void at::native::unrolled_elementwise_kernel<at::native::BinaryFunctor<float, float, float, at::native::AddFunctor<float>>, at::detail::Array<char *, (int)3>, OffsetCalculator<(int)2, unsigned int, (bool)0>, OffsetCalculator<(int)1, unsigned int, (bool)0>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, T1, T2, T3, T4, T5, T6)'] = [1, 'Add']
kernels_additional_info['void at::native::vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor<float, float, float, at::native::MulFunctor<float>>, at::detail::Array<char *, (int)2>>(int, T2, T3)'] = [1, 'Div']
kernels_additional_info['void at::native::vectorized_elementwise_kernel<(int)4, at::native::tanh_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 4)]::operator ()() const::[lambda(float) (instance 1)], at::detail::Array<char *, (int)2>>(int, T2, T3)'] = [1, 'Tanh']
'''

print(len(kernels))
#print(kernels[0])
labels = ['Kernel_Name', 'DRAM_Throughput(%)', 'Duration(ns)', 'Compute(SM)(%)',  'Block', 'Grid', 'Registers_Per_Thread', 'Static_shmem_per_block', 'Number_of_threads', 'Number_of_registers']



df_new = pd.DataFrame(kernels, columns=labels)
print(df_new)
df_new.to_csv(f'{pwd}/output_ncu_processed.csv')
