#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <cmath>
#include "cl/sum_cl.h"
#include "cl/max_cl.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

gpu::gpu_mem_32i prefix_sum(
        gpu::gpu_mem_32i &a_gpu,
        ocl::Kernel &ocl_prefix_sum_kernel,
        unsigned int n,
        unsigned int work_group_size) {

    gpu::gpu_mem_32i prefix_sums;
    prefix_sums.resizeN(n);

    gpu::gpu_mem_32i block_sums;
    unsigned int block_buffer_size = (n + work_group_size - 1) / work_group_size;
    block_sums.resizeN(block_buffer_size);

    ocl_prefix_sum_kernel.exec(gpu::WorkSize(work_group_size, n), a_gpu, a_gpu, prefix_sums, block_sums, 0, n);

    if (n <= work_group_size) {
        return prefix_sums;
    }

    gpu::gpu_mem_32i block_prefix_sums =
            prefix_sum(block_sums, ocl_prefix_sum_kernel, block_buffer_size, work_group_size);

    ocl_prefix_sum_kernel.exec(
            gpu::WorkSize(work_group_size, n), a_gpu, block_prefix_sums, prefix_sums, block_sums, 1, n);
    return prefix_sums;
}

int main(int argc, char **argv)
{
    int benchmarkingIters = 10;
    int max_n = (1 << 24);

    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result <<
            ")" << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            gpu::Device device = gpu::chooseGPUDevice(argc, argv);

            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();

            unsigned int work_group_size = 256;

            ocl::Kernel ocl_prefix_sum_kernel(
                    max_kernel, max_kernel_length, "prefix_sum", "-DWORK_GROUP_SIZE=" + to_string(work_group_size));
            ocl_prefix_sum_kernel.compile(false);

            ocl::Kernel ocl_max_kernel(
                    max_kernel, max_kernel_length, "calc_max", "-DWORK_GROUP_SIZE=" + to_string(work_group_size));
            ocl_max_kernel.compile(false);

            gpu::gpu_mem_32i a_gpu;
            a_gpu.resizeN(n);
            a_gpu.writeN(as.data(), n);

            timer t;
            for (int i = 0; i < benchmarkingIters; ++i) {
                unsigned int n1 = n;

                gpu::gpu_mem_32i gpu_old_buffer = prefix_sum(a_gpu, ocl_prefix_sum_kernel, n, work_group_size);

                while (n1 > 1) {
                    gpu::gpu_mem_32i gpu_buffer;
                    unsigned int buffer_size = (n1 + work_group_size - 1) / work_group_size;
                    gpu_buffer.resizeN(buffer_size);

                    ocl_max_kernel.exec(gpu::WorkSize(work_group_size, n1), gpu_old_buffer, gpu_buffer, n1);
                    n1 = (n1 + work_group_size - 1) / work_group_size;
                    gpu_old_buffer = gpu_buffer;
                }

                int max_sum = 0;
                gpu_old_buffer.readN(&max_sum, 1);
                max_sum = std::max(0, max_sum);

                EXPECT_THE_SAME(reference_max_sum, max_sum, "GPU result should be consistent!");

                t.nextLap();
            }

            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
