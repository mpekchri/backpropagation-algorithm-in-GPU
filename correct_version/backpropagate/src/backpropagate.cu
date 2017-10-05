/*
 ============================================================================
 Name        : backpropagate.cu
 Author      : Christophoros Bekos (mpekchri@auth.gr)
 Version     :
 Copyright   : @ copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>

#define work_per_block 100
#define threads_per_warp 32

#define threads_per_warp 32

__device__ void sigmoid(float& z) {
	z = 1.0 / (1.0 + exp(-(z)));
}

__device__ void hadamard_product_small(float* sh_a, float* sh_b, float* sh_res,
		int multiplier, int size, int mult) {
	int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	int block_id = blockIdx.x;
	// start the computations
	int cnt = 0;
	for (int i = thread_id * multiplier;
			i < thread_id * multiplier + multiplier; i++) {
		sh_res[i * mult] = sh_b[i] * sh_a[i] * ((int) (i < size));
		cnt++;
	}
	// result is stored in sh_b vector\
	//done
}

__device__ void array_sum_small(float* sha, float& result, int size,
		int start) {
	int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

	// start the computations
	for (int i = threads_per_warp; i < work_per_block; i = i * 2) {
		// switch 1 : even warps add their's neighbors contents
		switch ((int) floor(thread_id / (double) i) % 2) {
		case 0:
			// thread_id  % i == even
			// add the "more next vector"
			sha[thread_id] = sha[thread_id]
					+ sha[i + thread_id]
							* ((int) (start + thread_id + i < size));
			break;
		default:
			// thread_id  % i == odd
			// do nothing
			break;
		}
		__syncthreads();
		// switch2 : odd warps clean up their content
		switch ((int) floor(thread_id / (double) i) % 2) {
		case 0:
			// thread_id  % i == even
			// do nothing
			break;
		default:
			// thread_id  % i == odd
			// clean up
			sha[thread_id] = 0;
			//__syncthreads();
			break;
		}
		__syncthreads();
	}

	// loop ended, sha[0:threads_per_warp] got the sum
	if (thread_id == 0) {
		for (int i = 0; i < threads_per_warp; i++) {
			result = result + sha[i];
			sha[i] = 0;
		}
	}
}

__device__ void backpropagate_some_cols(float* result, int rows_per_block,
		int col_length, float* matrix, float* vector, int last_block, int size,
		float* sigm_der) {
	// README :
	// each block uses rows threads
	// each block modifies rows columns ( cols columns per block)
	// each thread modifies one column , column's length is col_length
	// cols : number of columns that this block will modify
	// one last block has less job to do, this one takes parameter last_block == 1
	// and size (after index exceeds size in last block, no computation must be made)

	int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	int block_id = blockIdx.x;

	extern __shared__ float shared[];
	float* temp = shared;
	float* m = &temp[rows_per_block];
	float* v = &m[col_length * rows_per_block];
	float* res = &v[col_length * rows_per_block];

	// move data in shared memory
	for (int i = thread_id * col_length;
			i < thread_id * col_length + col_length; i++) {
		m[i] = matrix[i];
	}
	v[thread_id] = 0;
	v[thread_id] = vector[thread_id] * (thread_id < col_length);

	__syncthreads();

	int cnt = 0;
	for (int i = thread_id * col_length;
			i < thread_id * col_length + col_length; i++) {
		m[i] = m[i] * v[cnt];
		cnt++;
	}
	__syncthreads();

	temp[thread_id] = 0;
	for (int i = thread_id * col_length;
			i < thread_id * col_length + col_length; i++) {
		temp[thread_id] += m[i];

	}
	__syncthreads();
	result[thread_id] = temp[thread_id] * sigm_der[thread_id];

}

__global__ void backpropagate(float* result, int rows_per_block, int col_length,
		float* matrix, float* vector, int last_block, int size,
		float* sigm_der) {
	int block_id = blockIdx.y * gridDim.x + blockIdx.x;
	int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	backpropagate_some_cols(&result[block_id * rows_per_block], rows_per_block,
			col_length, &matrix[block_id * rows_per_block], vector,
			(block_id == last_block), size,
			&sigm_der[block_id * rows_per_block]);

}

void initialize(float *data, unsigned size, float arg) {
	for (unsigned i = 0; i < size; ++i) {
		data[i] = arg;
	}
}

void cpu_backpropagate(float* d_L, int rows, int cols, float** d_new,
		float* sigm_der, float* w);

int main(void) {
	int rows = 783;
	int cols = 30;
	float *w = new float[rows * cols];
	float *d_old = new float[cols];
	float *delta = new float[rows];
	float *delta_gpu = new float[rows];
	float* sigm_der = new float[rows];
	float *m, *v, *new_delta, *sigm_der_gpu;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			w[i * cols + j] = 1.2;
		}
	}

	initialize(d_old, cols, 1.5);
	initialize(sigm_der, rows, 1.6);

	cudaMalloc((void**) &m, sizeof(float) * (rows * cols));
	cudaMalloc((void**) &v, sizeof(float) * cols);
	cudaMalloc((void**) &new_delta, sizeof(float) * rows);
	cudaMalloc((void**) &sigm_der_gpu, sizeof(float) * rows);
	cudaMemcpy(m, w, sizeof(float) * (rows * cols), cudaMemcpyHostToDevice);
	cudaMemcpy(v, d_old, sizeof(float) * cols, cudaMemcpyHostToDevice);
	cudaMemcpy(sigm_der_gpu, sigm_der, sizeof(float) * rows,
			cudaMemcpyHostToDevice);

	int numofthreads = work_per_block;
	int rows_per_block = numofthreads;
	int col_length = cols;
	int last_block = floor(rows / work_per_block);
	float cache = 11000 * sizeof(float);
	int num_of_blocks = floor(rows / work_per_block) + 1;
	int size_for_last_block = rows
			- floor(rows / work_per_block) * numofthreads;
	// printf("aaaa %d \n", num_of_blocks);
	// BACKPROPAGATE FOR 1 ITERATION

	// IN GPU
	//printf("sadfa %d ",size_for_last_block);
	backpropagate<<<num_of_blocks, rows_per_block, cache>>>(new_delta,
			rows_per_block, col_length, m, v, last_block, size_for_last_block,
			sigm_der_gpu);

	cudaDeviceSynchronize();
	cudaMemcpy(delta_gpu, new_delta, sizeof(float) * rows,
			cudaMemcpyDeviceToHost);

	// IN CPU
	cpu_backpropagate(d_old, rows, cols, &delta, sigm_der, w);

	// COMPARE RESULTS
	int success = 1;
	for (int i = 0; i < rows; i++) {
		// printf("kappa %f \n", delta[i]);
		if (delta[i] != delta_gpu[i]) {
			printf("ERROR in a, cpu = %f, gpu = %f\n", delta[i], delta_gpu[i]);
			success = 0;
		}
	}
	/* Free memory */
	cudaFree(new_delta);
	cudaFree(m);
	cudaFree(v);
	if (success) {
		printf("SUCCESS \n");
	}
	return 0;
}

float* hadamard_product(int size, float* a, float* b) {
	// returns the datamard product for vectors a and b
	// (return a.*b in matlab)
	// size = length of arrays a and b
	float* result = new float[size];
	for (int i = 0; i < size; i++) {
		result[i] = a[i] * b[i];
	}
	return result;
}

float* mull_backpropagate(int rows, int cols, float* matrix, float* vector) {
	// TESTED
	// returns "rows x 1" vector
	float* temp = NULL;
	float* res = new float[rows];
	for (int j = 0; j < rows; j++) {
		temp = hadamard_product(cols, &matrix[j * cols], vector);
		res[j] = 0;
		for (int i = 0; i < cols; i++) {
			res[j] += temp[i];
		}
		delete[] temp;
	}

	return res;
}

void cpu_backpropagate(float* d_L, int rows, int cols, float** d_new,
		float* sigm_der, float* w) {
	float* w_d;
	w_d = mull_backpropagate(rows, cols, w, d_L);
	d_new[0] = hadamard_product(rows, w_d, sigm_der);
	delete[] w_d;
}
