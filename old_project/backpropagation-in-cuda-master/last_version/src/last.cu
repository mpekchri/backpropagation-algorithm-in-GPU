/*
 ============================================================================
 Name        : last.cu
 Author      : christopher
 Version     :
 Copyright   : @ copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#define threads_per_block 256
#define threads_per_warp 32

//#include "/home/chris/Downloads/cuPrintf.cu"
//#include "/home/chris/Downloads/cuPrintf.cuh"

__device__ void hadamard_product_small(double* sh_a, double* sh_b,
		int multiplier, int rows) {
	int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

	// start the computations
	for (int i = thread_id * multiplier;
			i < thread_id * multiplier + multiplier; i++) {
		sh_b[i] = sh_b[i] * sh_a[i] * (i < rows);
	}
	// result is stored in sh_b vector\
	//done
}

__device__ void array_sum_small(double* sha, double& result, int multiplier,
		int rows, int start) {
	int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

	// start the computations
	for (int i = threads_per_warp; i < threads_per_block; i = i * 2) {
		// switch 1 : even warps add their's neighbors contents
		switch ((int) floor(thread_id / (double) i) % 2) {
		case 0:
			// thread_id  % i == even
			// add the "more next vector"
			sha[thread_id] = sha[thread_id]
					+ sha[i + thread_id] * (start + thread_id + i < rows);
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
		}
	}
}

__global__ void array_mult(double* matrix, double* vector, double* result,
		int rows, int cols_per_block, int multiplier, double* sigm) {

	double* a = &matrix[blockIdx.x * rows * cols_per_block];
	//result[0] = 0;
	extern __shared__ double shared[];
	double* sh_m = shared;
	double* sh_v = &sh_m[threads_per_block * multiplier];
	double* res = &sh_v[threads_per_block * multiplier];
	// thread_id*multiplier ews thread_id*multiplier+multiplier-1

	int thread_id = threadIdx.x;

	for (int c = 0; c < cols_per_block; c++) {
		// for each col that every block must deal with , do the following :

		// load from global to shared mem
		for (int i = thread_id * multiplier;
				i < thread_id * multiplier + multiplier; i++) {
			sh_m[i] = a[i + c * rows] * (i < rows);
		}
		for (int i = thread_id * multiplier;
				i < thread_id * multiplier + multiplier; i++) {
			sh_v[i] = vector[i + c * rows] * (i < rows);
		}
		__syncthreads();

		// find the hadamard product
		hadamard_product_small(sh_m, sh_v, multiplier, rows);
		__syncthreads();

		// initiallize shared vector res with zeros
		for (int i = thread_id * multiplier;
				i < thread_id * multiplier + multiplier; i++) {
			res[i] = 0;
		}
		__syncthreads();
		for (int i = 0; i < multiplier; i++) {
			array_sum_small(&sh_v[i * threads_per_block], res[i], multiplier,
					rows, i * threads_per_block);
		}
		__syncthreads();
		if (thread_id == 0) {
			for (int i = 1; i < multiplier; i++) {
				res[0] += res[i];
			}
			result[blockIdx.x * cols_per_block + c] = res[0]
					* sigm[blockIdx.x * cols_per_block + c];
		}
	}
}

using namespace std;
double getRandom(int min, int max);
double* matrix_vector_mull(int cols, int rows, double* matrix, double* vector);
int get_threads_per_cols(int cols);
int get_wSize_on_layer(int l, int* sizes);
int get_dSize_on_layer(int l, int* sizes);
double* hadamard_product(int size, double* a, double* b);
void backpropagate(double** delta, double** sigm_derivative,double** w, int* sizeOfLayers, int numOfLayers) ;

int main(void) {
	struct timeval t1, t2;
	double time, time_c, time_h;
	int num_of_layers = 4;
	int* sizes = new int[num_of_layers];
	cudaStream_t default_stream;
	cudaStreamCreate(&default_stream);
	sizes[0] = 9000;
	sizes[1] = 90;
	sizes[2] = 90;
	sizes[3] = 10;

	// seirial arrays
	double** w = new double*[num_of_layers - 1];
	double** delta = new double*[num_of_layers];
	double** sigm_der = new double*[num_of_layers];

	// cuda arrays
	double *w_c, *delta_c, *sigm_der_c;
	int w_length = 0, d_length = 0;
	w_length = get_wSize_on_layer(num_of_layers - 1, sizes);
	d_length = get_wSize_on_layer(num_of_layers, sizes);

	// gpu mem allocation
	cudaMalloc((void**) &w_c, sizeof(double) * w_length);
	cudaMalloc((void**) &delta_c, sizeof(double) * d_length);
	cudaMalloc((void**) &sigm_der_c, sizeof(double) * d_length);

	// host mem allocation
	for (int i = 0; i < num_of_layers - 1; i++) {
		w[i] = new double[sizes[i] * sizes[i + 1]];
		for (int j = 0; j < sizes[i] * sizes[i + 1]; j++) {
			w[i][j] = 1;
		}
	}
	for (int i = 0; i < num_of_layers; i++) {
		delta[i] = new double[sizes[i]];
		for (int j = 0; j < sizes[i]; j++) {
			delta[i][j] = 1;
		}
	}
	for (int i = 0; i < num_of_layers; i++) {
		sigm_der[i] = new double[sizes[i]];
		for (int j = 0; j < sizes[i]; j++) {
			sigm_der[i][j] = 0.5;
		}
	}
	// backpropagate requires only the delta[sizes[num_of_layers-1]] , so
	// we are not going to count the cudaMemcy's of the rest data,
	// simply because they can happen after cpu (host) updates the w's
	// and we got enough time from that point until we reach backpropagate function

	// suppose we do that way before we get close to backpropagation fucntion call
	// copy w in cuda
	for (int i = 0; i < num_of_layers - 1; i++) {
		cudaMemcpyAsync(&w_c[get_wSize_on_layer(i, sizes)], w[i],
				sizeof(double) * sizes[i] * sizes[i + 1],
				cudaMemcpyHostToDevice, default_stream);
	}
	// copy sigm_der in cuda
	for (int i = 0; i < num_of_layers; i++) {
		cudaMemcpyAsync(&sigm_der_c[get_dSize_on_layer(i, sizes)], sigm_der[i],
				sizeof(double) * sizes[i], cudaMemcpyHostToDevice,
				default_stream);
	}

	// copies done , wait for steam 0 (default) to compute all copies (as we said, we do not count them)
	cudaStreamSynchronize(default_stream);

	// now we may procced to backpropagation algorithm
	int multiplier = 0;

	gettimeofday(&t1, 0);

	// step 1 : copy the delta of the last layer into gpu
	// cpu commands : delta[numOfLayers - 1] = d_L;
	cudaMemcpyAsync(&delta_c[get_dSize_on_layer(num_of_layers - 1, sizes)],
			delta[num_of_layers - 1], sizeof(double) * sizes[num_of_layers - 1],
			cudaMemcpyHostToDevice, default_stream);

	// step 2
	int bl = 0;
	for (int i = num_of_layers - 2; i >= 0; i--) {
		// w_d = matrix_vector_mull(sizeOfLayers[i + 1], sizeOfLayers[i + 2], w[i], delta[i + 1]);

		if(i>0){
			multiplier = get_threads_per_cols(sizes[i]);// multiplier = get_threads_per_cols(cols);
			bl = sizes[i];
		}else{
			multiplier = get_threads_per_cols(sizes[i+1]);
			bl = sizes[i+1];
		}
		array_mult<<<bl, threads_per_block,
				sizeof(double) * (3 * threads_per_block * multiplier),
				default_stream>>>(&w_c[get_wSize_on_layer(i, sizes)],
				&delta_c[get_dSize_on_layer(i + 1, sizes)],
				&delta_c[get_dSize_on_layer(i, sizes)], sizes[i + 1], 1,
				multiplier, &sigm_der_c[get_dSize_on_layer(i, sizes)]);

		// delta[i] = hadamard_product(sizeOfLayers[i + 1], w_d, sigm_derivative[i]);
		cudaStreamSynchronize(default_stream);
		cudaMemcpyAsync(delta[i], &delta_c[get_dSize_on_layer(i, sizes)],
				sizeof(double) * sizes[i], cudaMemcpyDeviceToHost,
				default_stream);

	}
	// wait until the last copy is completed
	cudaStreamSynchronize(default_stream);
	// done
	gettimeofday(&t2, 0);
	time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec)
			/ 1000.0;
	cout << "Parallel time is " << time << " millisec \n";
	time_c = time;
	// retrieve data back to cpu memory for debbugin reasons

	cout<< "cuda results : \n";
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 3; j++) {
			cout<< delta[i][j] << " ";
		}
	 }
	cout<< "\n";

	// now the serial code
	gettimeofday(&t1, 0);
	backpropagate(delta, sigm_der,w,sizes,num_of_layers);


	gettimeofday(&t2, 0);
	time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec)
			/ 1000.0;
	cout << "serial time is " << time << " millisec \n";
	time_h = time;

	cout<< "cpu results : \n";
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 3; j++) {
			 cout<< delta[i][j] << " ";
		}
    }
	cout<< "\n";
	cout << "accelaration is " << (time_h-time_c)*100 << " % \n";
	cout << "SUCCESS epitelous";
	return 0;
}

void backpropagate(double** delta, double** sigm_derivative,
		double** w, int* sizeOfLayers, int numOfLayers) {
	double* w_d;
	for (int i = numOfLayers - 2; i >= 0; i--) {
		w_d = matrix_vector_mull(sizeOfLayers[i], sizeOfLayers[i + 1], w[i],delta[i + 1]);
		delta[i] = hadamard_product(sizeOfLayers[i], w_d, sigm_derivative[i]);
		delete[] w_d;
	}

}

double* hadamard_product(int size,double* a, double* b) {
	// returns the datamard product for vectors a and b
	// (return a.*b in matlab)
	// size = length of arrays a and b
	double* result = new double[size];
	for (int i = 0; i < size; i++) {
		result[i] = a[i] * b[i];
	}
	return result;
}

double* matrix_vector_mull(int cols, int rows, double* matrix, double* vector) {
	// TESTED
	// returns "cols x 1" vector
	double* temp = NULL ;
	double* res = new double[cols];
	for(int j=0; j<cols; j++){
		temp = new double[rows] ;
		for(int i=0; i<rows; i++){
			temp[i] = matrix[i*cols+j];
		}
		temp = hadamard_product(rows,temp,vector);
		res[j] = 0;
		for(int i=0; i<rows; i++){
			res[j] += temp[i];
		}
		delete[] temp;
	}
	return res;
}

double getRandom(int min, int max) {
	return (((max - min) * ((double) rand() / (double) RAND_MAX) + min) * 100)
			/ 100;
}

int get_threads_per_cols(int cols) {
	if (cols < threads_per_block) {
		return 1;
	}
	int res = floor(cols / (double) threads_per_block);
	if (cols / (double) threads_per_block
			- floor(cols / (double) threads_per_block)) {
		res++;
	}
	return res;
}

int get_wSize_on_layer(int l, int* sizes) {
	int res = 0;
	for (int i = 0; i < l; i++) {
		res += sizes[i] * sizes[i + 1];
	}
	return res;
}

int get_dSize_on_layer(int l, int* sizes) {
	int res = 0;
	for (int i = 0; i < l; i++) {
		res += sizes[i];
	}
	return res;
}

