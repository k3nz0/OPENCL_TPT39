__kernel void matrix_mult(__global const float *x, 
                        __global const float *y, 
                        __global float *restrict z,
                        const int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    float res = 0.0;
    for(int k = 0; k < N; k++) {
        res += x[i * N + k] * y[j + k * N];
    }
    z[i * N + j] = res;
}

