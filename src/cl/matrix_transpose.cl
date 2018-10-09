#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#define WORK_GROUP_SIZE 16
#endif

#line 7

__kernel void matrix_transpose(__global const float *a, __global float *b, unsigned int n, unsigned int m) {
    __local float mem[WORK_GROUP_SIZE][WORK_GROUP_SIZE];
    __local float mem_t[WORK_GROUP_SIZE][WORK_GROUP_SIZE];

    unsigned int local_id_x = get_local_id(0);
    unsigned int local_id_y = get_local_id(1);

    unsigned int global_id_x = get_global_id(0);
    unsigned int global_id_y = get_global_id(1);

    {
        unsigned int offset_x = global_id_x - local_id_x;
        unsigned int offset_y = global_id_y - local_id_y;

        unsigned int global_x = offset_x + local_id_y;
        unsigned int global_y = offset_y + local_id_x;

        if (global_x < n && global_y < m) {
            mem[local_id_y][local_id_x] = a[global_x * m + global_y];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int y = (local_id_y + local_id_x) % WORK_GROUP_SIZE;
    mem_t[y][local_id_x] = mem[local_id_x][y];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_id_y < m && global_id_x < n) {
        b[global_id_y * n + global_id_x] = mem_t[local_id_y][local_id_x];
    }
}