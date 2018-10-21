#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

gpu::gpu_mem_32u prefix_sum(
        gpu::gpu_mem_32u &a_gpu,
        ocl::Kernel &ocl_prefix_sum_kernel,
        unsigned int n,
        unsigned int work_group_size) {

    gpu::gpu_mem_32u prefix_sums;
    prefix_sums.resizeN(n);

    gpu::gpu_mem_32u block_sums;
    unsigned int block_buffer_size = (n + work_group_size - 1) / work_group_size;
    block_sums.resizeN(block_buffer_size);

    ocl_prefix_sum_kernel.exec(
            gpu::WorkSize(work_group_size, n), a_gpu, a_gpu, prefix_sums, block_sums, 0, n);

    if (n <= work_group_size) {
        return prefix_sums;
    }

    gpu::gpu_mem_32u block_prefix_sums =
            prefix_sum(block_sums, ocl_prefix_sum_kernel, block_buffer_size, work_group_size);

    ocl_prefix_sum_kernel.exec(
            gpu::WorkSize(work_group_size, n), a_gpu, block_prefix_sums, prefix_sums, block_sums, 1, n);
    return prefix_sums;
}


void radix_sort(
        gpu::gpu_mem_32u &a_gpu,
        ocl::Kernel &prefix_sum_kernel,
        ocl::Kernel &move_numbers_kernel,
        ocl::Kernel &fill_bit_count_kernel,
        unsigned int n,
        unsigned int work_group_size,
        unsigned int radix_bits) {

    unsigned int global_work_size = (n + work_group_size - 1) / work_group_size * work_group_size;

    gpu::gpu_mem_32u count_buf;
    count_buf.resizeN(n * (1 << radix_bits));

    gpu::gpu_mem_32u b_gpu;
    b_gpu.resizeN(n);

    for (int start = 0; start < sizeof(unsigned int) * 8; start += radix_bits) {
        fill_bit_count_kernel.exec(
                gpu::WorkSize(work_group_size, global_work_size), a_gpu, count_buf, start, n);

        gpu::gpu_mem_32u sum = prefix_sum(count_buf, prefix_sum_kernel, n * (1 << radix_bits), work_group_size);

        move_numbers_kernel.exec(gpu::WorkSize(work_group_size, global_work_size), a_gpu, b_gpu, sum, start, n);

        a_gpu.swap(b_gpu);
    }

}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 1;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    {
        unsigned int work_group_size = 256;
        unsigned int radix_bits = 2;

        std::string compile_flags =
                "-DWORK_GROUP_SIZE=" + to_string(work_group_size) + " -DRADIX_BITS=" + to_string(radix_bits);
        ocl::Kernel move_numbers(radix_kernel, radix_kernel_length, "move_numbers", compile_flags);
        ocl::Kernel prefix_sum(
                radix_kernel, radix_kernel_length, "prefix_sum", compile_flags);
        ocl::Kernel fill_bit_count(radix_kernel, radix_kernel_length, "fill_bit_count", compile_flags);

        move_numbers.compile();
        prefix_sum.compile();
        fill_bit_count.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            radix_sort(as_gpu, prefix_sum, move_numbers, fill_bit_count, n, work_group_size, radix_bits);
//            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
//            radix.exec(gpu::WorkSize(workGroupSize, global_work_size),
//                       as_gpu, n);
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
