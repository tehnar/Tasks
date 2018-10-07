#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);

        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        unsigned int work_group_size = 256;
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum", "-DWORK_GROUP_SIZE=" + to_string(work_group_size));

        kernel.compile(false);

        gpu::gpu_mem_32u gpu_a;
        gpu_a.resizeN(n);
        gpu_a.writeN(as.data(), n);

        timer t;
        for (int i = 0; i < benchmarkingIters; ++i) {
            unsigned int n1 = n;

            gpu::gpu_mem_32u gpu_old_buffer = gpu_a;

            while (n1 >= work_group_size) {
                gpu::gpu_mem_32u gpu_buffer;
                unsigned int buffer_size = (n1 + work_group_size - 1) / work_group_size;
                gpu_buffer.resizeN(buffer_size);

                kernel.exec(gpu::WorkSize(work_group_size, n1), gpu_old_buffer, gpu_buffer, n1);
                n1 = (n1 + work_group_size - 1) / work_group_size;
                gpu_old_buffer = gpu_buffer;
            }

            std::vector<unsigned int> sums(n1);
            gpu_old_buffer.readN(sums.data(), n1);
            unsigned int sum = 0;
            for (int j = 0; j < n1; j++) {
                sum += sums[j];
            }

            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");

            t.nextLap();
        }

        size_t flopsInLoop = 10;
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
}