/*
 ============================================================================
 Name        : mull_forward.cu
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

__device__ void sigmoid(float& z){
	z = 1.0 / (1.0 + exp(-(z))) ;
}

__device__ void hadamard_product_small(float* sh_a, float* sh_b, float* sh_res,
		int multiplier, int size) {
	int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	// start the computations
	for (int i = thread_id * multiplier;
			i < thread_id * multiplier + multiplier; i++) {
		sh_res[i] = sh_b[i] * sh_a[i] * ((int)(i < size));
	}
	// result is stored in sh_b vector\
	//done
}

__device__ void array_sum_small(float* sha, float& result,
		int size, int start) {
	int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

	// start the computations
	for (int i = threads_per_warp; i < work_per_block; i = i * 2) {
		// switch 1 : even warps add their's neighbors contents
		switch ((int) floor(thread_id / (double) i) % 2) {
		case 0:
			// thread_id  % i == even
			// add the "more next vector"
			sha[thread_id] = sha[thread_id]
					+ sha[i + thread_id] * ((int)(start + thread_id + i < size));
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

__device__ void mull_feedforward_one_col(float* result, int rows, int cols,
		float* matrix, float* vector, int multiplier,int size,float bias,float* sigm_der) {

	int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	int block_id = blockIdx.y * gridDim.x + blockIdx.x;
	extern __shared__ float shared[];
	float* temp = shared;
	float* m = &temp[rows * multiplier ];
	float* v = &m[rows * multiplier];
	float* res = &v[rows * multiplier];

	for (int i = thread_id * multiplier;
			i < thread_id * multiplier + multiplier; i++) {
		m[i] = matrix[i]*((i<size));

	}
	for (int i = thread_id * multiplier;
			i < thread_id * multiplier + multiplier; i++) {
		v[i] = vector[i]*((i<size));
	}
	for (int i = thread_id * multiplier;
			i < thread_id * multiplier + multiplier; i++) {
		res[i] = 0.0;
	}
	for (int i = thread_id * multiplier;
				i < thread_id * multiplier + multiplier; i++) {
			temp[i] = 0.0;
	}
	__syncthreads();

	hadamard_product_small(m, v, temp, multiplier, size);
	__syncthreads();

	for (int i = multiplier-1; i >=0; i--) {
		array_sum_small(&temp[i*work_per_block],res[0], size, (i * work_per_block));
		__syncthreads();
	}

	if (thread_id == 0) {
		float tmp = (res[thread_id] + bias);
		sigmoid(tmp);
		result[block_id] = tmp;
		sigm_der[block_id] = tmp*(1-tmp);
	}

}

__global__ void feedforward(float* result, int rows, int cols,
		float* matrix, float* vector, int multiplier,int size,float* biases,float* sigm_der){
	int block_id = blockIdx.y * gridDim.x + blockIdx.x;
	mull_feedforward_one_col(result,rows,cols,&matrix[block_id*size],vector,multiplier,size,biases[block_id],sigm_der);
}


void initialize(float *data, unsigned size, float arg) {
	for (unsigned i = 0; i < size; ++i) {
		data[i] = arg;
	}
}

float* mull_feedforward(int rows, int cols, float* matrix, float* vector);
void cpu_feedforward(float* a_old,int rows, int cols,float** a_new,float* w,float* b,float* sigm_der_result) ;

int main(void) {
	int rows = 783;
	int cols = 30;
	float *w = new float[rows*cols];
	float *a_old = new float[rows];
	float *b = new float[cols];
	float *res = new float[cols];
	float* sigm_der_result = new float[cols];
	float* sigm_der_gpu = new float[cols];
	float *m, *v, *a_next,*sigm_der,*biases;
	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			w[i*cols+j] = 1.0;
			// printf("index is %d , v = %0.6f \n",i*cols+j,matrix[i*cols+j]);
		}
	}
	initialize(a_old, rows, 2.0);
	initialize(b, cols, 10.5);

	cudaMalloc((void**) &m, sizeof(float) * (rows * cols));
	cudaMalloc((void**) &v, sizeof(float) * rows);
	cudaMalloc((void**) &a_next, sizeof(float) * cols);
	cudaMalloc((void**) &sigm_der, sizeof(float) * cols);
	cudaMalloc((void**) &biases, sizeof(float) * cols);
	cudaMemcpy(m, w, sizeof(float) * (rows * cols), cudaMemcpyHostToDevice);
	cudaMemcpy(v, a_old, sizeof(float) * rows, cudaMemcpyHostToDevice);
	cudaMemcpy(biases, b, sizeof(float) * cols, cudaMemcpyHostToDevice);

	int numofthreads = work_per_block;
	int size = rows;
	int multiplier = floor(rows/work_per_block)+1;
	if(rows<work_per_block){
		multiplier = 1;
	}
	float cache = 11000 * sizeof(float);

	// FEEDFORWARD FOR 1 ITERATION

	// IN GPU
	feedforward<<<cols, numofthreads, cache>>>(a_next, work_per_block, cols, m, v,
			multiplier,size,biases,sigm_der);
	cudaDeviceSynchronize();
	cudaMemcpy(res, a_next, sizeof(float) * cols, cudaMemcpyDeviceToHost);
	cudaMemcpy(sigm_der_gpu, sigm_der, sizeof(float) * cols, cudaMemcpyDeviceToHost);

	// IN CPU
	float* a_new = new float[cols];
	cpu_feedforward(a_old,rows,cols,&a_new,w,b,sigm_der_result);

	// COMPARE RESULTS
	int success = 1;
	for (int i = 0; i < cols; i++) {
		// printf("kappa %f \n", res[i]);
		if(a_new[i]!=res[i]){
			printf("ERROR in a, cpu = %f, gpu = %f\n",a_new[i], res[i]);
			success = 0;
		}
		if(sigm_der_gpu[i]!=sigm_der_result[i]){
			printf("ERROR in a, cpu = %f, gpu = %f\n",sigm_der_result[i], sigm_der_gpu[i]);
			success = 0;
		}
	}
	/* Free memory */
	cudaFree(a_next);
	cudaFree(m);
	cudaFree(v);
	cudaFree(biases);
	cudaFree(sigm_der);
	if(success){
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

float* mull_feedforward(int rows, int cols, float* matrix, float* vector) {
	// TESTED
	// returns "cols x 1" vector
	float* temp = NULL;
	float* res = new float[cols];
	for (int j = 0; j < cols; j++) {
		temp = hadamard_product(rows, &matrix[j * rows], vector);
		res[j] = 0;
		for (int i = 0; i < rows; i++) {
			res[j] += temp[i];
		}
		delete[] temp;
	}
	return res;
}


void vector_add(int size, float* a, float* b) {
	for (int i = 0; i < size; i++) {
		a[i] += b[i];
	}
}

float sigm(float z) {
	return 1.0 / (1.0 + exp(-z));
}

void sigmoid(float** z, int size) {
	for (int i = 0; i < size; i++) {
		(*z)[i] = sigm(((*z)[i]));
	}
}

float* compute_z(float* a, float* w, float* b, int rows, int cols) {
	float* result = mull_feedforward(rows, cols, w, a);
	vector_add(cols, result, b);
	return result;
}

void compute_sigm_der(float* a, float* result, int size) {
	for (int i = 0; i < size; i++) {
		result[i] = a[i] * (1 - a[i]);
	}
}

void cpu_feedforward(float* a_old,int rows, int cols,float** a_new,float* w,float* b,float* sigm_der_result) {
	a_new[0] = compute_z(a_old, w, b, rows, cols);
	sigmoid(&a_new[0], cols);
	compute_sigm_der(a_new[0], sigm_der_result, cols);
}
