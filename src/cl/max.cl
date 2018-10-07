#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void calc_max(__global int *a, __global int *results, unsigned int n) {
    __local int mem[2 * WORK_GROUP_SIZE];
    unsigned int local_id = get_local_id(0);
    unsigned int global_id = get_global_id(0);

    printf("local_id=%d\n", local_id);
    int old_offset = 0;
    int offset = WORK_GROUP_SIZE;

    if (global_id < n) {
        mem[local_id] = a[global_id];
    } else {
        mem[local_id] = -1000000000;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    printf("local_id=%d\n", local_id);

    for (int step = WORK_GROUP_SIZE / 2; step > 0; step /= 2) {
        if (local_id < step) {
            mem[offset + local_id] = max(mem[old_offset + local_id], mem[old_offset + local_id + step]);
        }
        old_offset = offset;
        offset += step;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!local_id) {
        results[global_id / WORK_GROUP_SIZE] = mem[offset - 1];
    }
}

__kernel void prefix_sum(
        __global int *a,
        __global int *block_sums,
        __global int *output,
        __global int *block_output,
        int add_block_sums,
        unsigned int n) {
    unsigned int local_id = get_local_id(0);
    unsigned int global_id = get_global_id(0);

    __local int mem1[WORK_GROUP_SIZE];
    __local int result1[WORK_GROUP_SIZE];

    __local int *mem = mem1;
    __local int *result = result1;

    printf("local_id=%d\n", local_id);

    if (global_id < n) {
        mem[local_id] = result[local_id] = a[global_id];
    } else {
        mem[local_id] = result[local_id] = 0;
    }

    if (add_block_sums && local_id == 0 && global_id && global_id < n) {
        int prev_value = block_sums[global_id / WORK_GROUP_SIZE - 1];
        mem[0] += prev_value;
        result[0] += prev_value;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    printf("local_id=%d\n", local_id);

    for (unsigned int step = 1; step < WORK_GROUP_SIZE; step *= 2) {
        if (local_id >= step) {
            result[local_id] = mem[local_id] + mem[local_id - step];
        } else {
            result[local_id] = mem[local_id];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        __local int *tmp = mem;
        mem = result;
        result = tmp;
    }

    if (global_id < n) {
        output[global_id] = mem[local_id];
    }

    if (local_id == 0 && global_id < n) {
        block_output[global_id / WORK_GROUP_SIZE] = mem[WORK_GROUP_SIZE - 1];
    }
}

