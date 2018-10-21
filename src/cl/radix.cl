#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#define RADIX_BITS 4
#endif

#line 7

__kernel void move_numbers(
        __global unsigned int* a,
        __global unsigned int* b,
        __global unsigned int *number_count,
        unsigned int start_bit,
        unsigned int n) {
    unsigned int global_id = get_global_id(0);
    if (global_id < n) {
        unsigned int x = a[global_id];
        unsigned int val = (x >> start_bit) & ((1 << RADIX_BITS) - 1);
        unsigned int move_pos = number_count[val * n + global_id] - 1;
        b[move_pos] = x;
    }
}


__kernel void fill_bit_count(
        __global unsigned int *a,
        __global unsigned int *number_count_output,
        unsigned int start_bit,
        unsigned int n) {
    unsigned int global_id = get_global_id(0);
    if (global_id < n) {
        unsigned int x = a[global_id];
        unsigned int val = (x >> start_bit) & ((1 << RADIX_BITS) - 1);
        for (int i = 0; i < (1 << RADIX_BITS); i++) {
            number_count_output[i * n + global_id] = i == val;
        }
    }
}

__kernel void prefix_sum(
        __global unsigned int *a,
        __global unsigned int *block_sums,
        __global unsigned int *output,
        __global unsigned int *block_output,
        int add_block_sums,
        unsigned int n) {
    unsigned int local_id = get_local_id(0);
    unsigned int global_id = get_global_id(0);

    __local unsigned int mem1[WORK_GROUP_SIZE];
    __local unsigned int result1[WORK_GROUP_SIZE];

    __local unsigned int *mem = mem1;
    __local unsigned int *result = result1;

    if (global_id < n) {
        mem[local_id] = result[local_id] = a[global_id];
    } else {
        mem[local_id] = result[local_id] = 0;
    }

    if (add_block_sums && local_id == 0 && global_id && global_id < n) {
        unsigned int prev_value = block_sums[global_id / WORK_GROUP_SIZE - 1];
        mem[0] += prev_value;
        result[0] += prev_value;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int step = 1; step < WORK_GROUP_SIZE; step *= 2) {
        if (local_id >= step) {
            result[local_id] = mem[local_id] + mem[local_id - step];
        } else {
            result[local_id] = mem[local_id];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        __local unsigned int *tmp = mem;
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
