// write convolution here and use it for filters
__kernel void convolve(__global const float *image, 
                        __global const float *filter,
                        __global float *restrict out)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int rows = get_global_size(0);
    int cols = get_global_size(1);

	int kernel_rows = 3;
	int kernel_cols =  3;

    int dx[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    int dy[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};

    float sum = 0.0;
    for(int k = 0; k < 9; k++) {
        int x = i + dx[k];
        int y = j + dy[k];
        float d = 0.0;
        if(x >= 0 && x < rows && y >= 0 && y < cols) {
            d = image[x * cols + y]; 
        }
        sum += d * filter[ (dx[k] + 1) * kernel_cols + (dy[k] + 1)];
    } 
    out[i * cols + j] = sum;
}
