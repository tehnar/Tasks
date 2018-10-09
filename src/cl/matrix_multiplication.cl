#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#define WORK_GROUP_SIZE 16
#endif

#line 7

__kernel void matrix_multiplication(
        __global const float *a, // n * m
        __global const float *b, // m * k
        __global float *c, // n * k
        unsigned int n,
        unsigned int m,
        unsigned int k) {

    unsigned int local_id_x = get_local_id(0);
    unsigned int local_id_y = get_local_id(1);

    unsigned int global_id_x = get_global_id(0);
    unsigned int global_id_y = get_global_id(1);

    __local float mem_a[WORK_GROUP_SIZE][WORK_GROUP_SIZE];
    __local float mem_b[WORK_GROUP_SIZE][WORK_GROUP_SIZE];
    __local float mem_b_t[WORK_GROUP_SIZE][WORK_GROUP_SIZE];

    float dot_product = 0;

    for (unsigned int for_x = 0; for_x < m; for_x += WORK_GROUP_SIZE) {
        {
            unsigned int offset_x = global_id_x - local_id_x;
            unsigned int offset_y = global_id_y - local_id_y;

            unsigned int global_x = offset_x + local_id_y;
            unsigned int global_bx = for_x + local_id_y;

            unsigned int global_y = offset_y + local_id_x;
            unsigned int global_ay = for_x + local_id_x;

            if (global_x < n && global_ay < m) {
                mem_a[local_id_y][local_id_x] = a[global_x * m + global_ay];
            }
            if (global_bx < m && global_y < k) {
                mem_b[local_id_y][local_id_x] = b[global_bx * k + global_y];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        int y = (local_id_y + local_id_x) % WORK_GROUP_SIZE;
        mem_b_t[y][local_id_x] = mem_b[local_id_x][y];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < WORK_GROUP_SIZE; i++) {
            dot_product += mem_a[local_id_x][i] * mem_b_t[local_id_y][i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_id_x < n && global_id_y < k) {
        c[global_id_x * k + global_id_y] = dot_product;
    }

}
