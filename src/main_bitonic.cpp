#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/bitonic_cl.h"

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

// Only arrays with length of power of 2 are supported
void bitonic_sort(
        ocl::Kernel &small_array_kernel,
        ocl::Kernel &large_array_kernel,
        const gpu::gpu_mem_32f &mem,
        unsigned int n,
        unsigned int work_group_size) {

    unsigned int global_work_size = (n + work_group_size - 1) / work_group_size * work_group_size;

    unsigned int size = 2;

    unsigned int max_size = 2;
    while (max_size < n) max_size *= 2;

    while (size <= max_size) {
        unsigned int cur_size = size / 2;

        while (2 * cur_size > work_group_size) {
            large_array_kernel.exec(gpu::WorkSize(work_group_size, global_work_size), mem, cur_size, size, n);
            cur_size /= 2;
        }

        small_array_kernel.exec(gpu::WorkSize(work_group_size, global_work_size), mem, cur_size, size, n);
        size *= 2;
    }
}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<float> cpu_sorted;
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

    gpu::gpu_mem_32f as_gpu;
    as_gpu.resizeN(n);

    {
        unsigned int work_group_size = 256;
        ocl::Kernel bitonic_small_array_kernel(
                bitonic_kernel, bitonic_kernel_length, "bitonic_small_array",
                "-DWORK_GROUP_SIZE=" + to_string(work_group_size));
        ocl::Kernel bitonic_large_array_kernel(
                bitonic_kernel, bitonic_kernel_length, "bitonic_large_array",
                "-DWORK_GROUP_SIZE=" + to_string(work_group_size));

        bitonic_small_array_kernel.compile();
        bitonic_large_array_kernel.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            bitonic_sort(bitonic_small_array_kernel, bitonic_large_array_kernel, as_gpu, n, work_group_size);
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    for (int i = 0; i < n - 1; ++i) {
        if (as[i] > as[i + 1]) {
            printf("%d %f %f\n", i, as[i], as[i + 1]);
        }
//        printf("%f ", as[i]);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
