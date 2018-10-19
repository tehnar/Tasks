#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#define WORK_GROUP_SIZE 256
#endif

#line 7

#define READ_TO_LOCAL_MEM() ;

__kernel void bitonic_small_array(__global float* as, unsigned int cur_size, unsigned int size, unsigned int n) {
    unsigned int local_id = get_local_id(0);
    unsigned int global_id = get_global_id(0);

    __local float mem[WORK_GROUP_SIZE];

    if (global_id < n) {
        mem[local_id] = as[global_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int is_sort_ascending = global_id % (2 * size) < size;

    while (cur_size >= 1) {

        if (global_id % (2 * cur_size) < cur_size && global_id + cur_size < n) {
            float a = mem[local_id];
            float b = mem[local_id + cur_size];
            if ((a < b) == is_sort_ascending) {
            } else {
                mem[local_id] = b;
                mem[local_id + cur_size] = a;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        cur_size /= 2;
    }

    if (global_id < n) {
        as[global_id] = mem[local_id];
    }
}


__kernel void bitonic_large_array(__global float* as, unsigned int cur_size, unsigned int size, unsigned int n) {
    unsigned int global_id = get_global_id(0);

    int is_sort_ascending = global_id % (2 * size) < size;

    if (global_id % (2 * cur_size) < cur_size && global_id + cur_size < n) {
        float a = as[global_id];
        float b = as[global_id + cur_size];
        if ((a < b) == is_sort_ascending) {
        } else {
            as[global_id] = b;
            as[global_id + cur_size] = a;
        }
    }
}
