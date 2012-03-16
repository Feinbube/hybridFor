/**
 * Average calculation for given array slice.
 */
__kernel void reduce(__global float *d_Result, __global float *d_Input, int N){
    const int tid = get_global_id(0);
    const int threadN = get_global_size(0);

    float sum = 0;
	int count = 0;

    for(int pos = tid; pos < N; pos += threadN) {
        sum += d_Input[pos];
		count++;
	}

    d_Result[tid] = sum / count;
}
