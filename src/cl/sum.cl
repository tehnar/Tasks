#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum(__global unsigned int *a, __global unsigned int *results, unsigned int n) {
    unsigned int local_id = get_local_id(0);
    unsigned int global_id = get_global_id(0);

    __local unsigned int mem[2 * WORK_GROUP_SIZE];

    int old_offset = 0;
    int offset = WORK_GROUP_SIZE;

    if (global_id < n) {
        mem[local_id] = a[global_id];
    } else {
        mem[local_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int step = WORK_GROUP_SIZE / 2; step > 0; step /= 2) {
        if (local_id < step) {
            mem[offset + local_id] = mem[old_offset + local_id] + mem[old_offset + local_id + step];
        }
        old_offset = offset;
        offset += step;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!local_id) {
        results[global_id / WORK_GROUP_SIZE] = mem[offset - 1];
    }
}
