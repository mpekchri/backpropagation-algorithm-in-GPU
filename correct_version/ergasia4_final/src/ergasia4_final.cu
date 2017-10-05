/*
 ============================================================================
 Name        : ergasia4_final.cu
 Author      : Christophoros Bekos (mpekchri@auth.gr)
 Version     :
 Copyright   : @ 
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define threads_per_warp 32
#define num_of_threads 256

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
	for (int i = threads_per_warp; i < num_of_threads; i = i * 2) {
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
		array_sum_small(&temp[i*num_of_threads],res[0], size, (i * num_of_threads));
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

void cpu_feedforward(float* a_old, int rows, int cols, float** a_new, float* w,
		float* b, float* sigm_der_result);
void train(int num_of_layers, int* s, float** w, float** b, float** alfa,
		float** delta, float** sigm_derivative);
float getRandom(int min, int max);
float* transformOutput(int output, int size);
float* cost_derivative(float* a, float* y, int size);
float* mull_feedforward(int rows, int cols, float* matrix, float* vector);
float* hadamard_product(int size, float* a, float* b);
void cpu_backpropagate(float* d_L, int rows, int cols, float** d_new,
		float* sigm_der, float* w);
void cuda_train(int num_of_layers, int* s, float** w, float** b, float** alfa,
		float** delta, float** sigm_derivative);

int main(void) {
	// SECTION 1 :
	// define network's size :
	int num_of_layers = 3;
	int* s = new int[num_of_layers];	// size of layers
	s[0] = 784;
	s[1] = 30;
	s[2] = 10;

	// SECTION 2 :
	// define network's structures
	float **w,**gpu_w;
	float **b,**gpu_b, **sigm_derivative,**gpu_sigm_derivative, **delta,**gpu_delta, **alfa,**gpu_alfa;
	//float **c_w, **c_b;

	w = new float*[num_of_layers];
	gpu_w = new float*[num_of_layers];
	//c_w = new float*[num_of_layers];
	b = new float*[num_of_layers];
	gpu_b = new float*[num_of_layers];
	//c_b = new float*[num_of_layers];
	delta = new float*[num_of_layers];
	sigm_derivative = new float*[num_of_layers];
	alfa = new float*[num_of_layers];
	gpu_delta = new float*[num_of_layers];
	gpu_sigm_derivative = new float*[num_of_layers];
	gpu_alfa = new float*[num_of_layers];

	alfa[0] = new float[s[0]];
	cudaMalloc((void**) &gpu_alfa[0], sizeof(float) * (s[0]));
	w[0] = NULL;
	b[0] = NULL;
	gpu_w[0] = NULL;
	gpu_b[0] = NULL;
	//c_w[0] = NULL;
	//c_b[0] = NULL;
	sigm_derivative[0] = NULL;
	delta[0] = NULL;
	for (int i = 1; i < num_of_layers; i++) {
		w[i] = new float[s[i - 1] * s[i]];
		cudaMalloc((void**) &gpu_w[i], sizeof(float) * (s[i-1]*s[i]));
		//c_w[i] = new float[s[i - 1] * s[i]];
		sigm_derivative[i] = new float[s[i]];
		cudaMalloc((void**) &gpu_sigm_derivative[i], sizeof(float) * (s[i]));
		b[i] = new float[s[i]];
		cudaMalloc((void**) &gpu_b[i], sizeof(float) * (s[i]));
		//c_b[i] = new float[s[i]];
		delta[i] = new float[s[i]];
		cudaMalloc((void**) &gpu_delta[i], sizeof(float) * (s[i]));
		alfa[i] = new float[s[i]];
		cudaMalloc((void**) &gpu_alfa[i], sizeof(float) * (s[i]));
	}
	for (int i = 1; i < num_of_layers; i++) {
		for (int j = 0; j < s[i]; j++) {
			b[i][j] = 1;
		}
	}
	for (int i = 1; i < num_of_layers; i++) {
		for (int j = 0; j < s[i - 1] * s[i]; j++) {
			w[i][j] = 0.5;
		}
	}

	// SECTION 3 :
	// Cuda initial data transfer
	cudaStream_t default_stream;
	cudaStreamCreate(&default_stream);

	for (int i = 1; i < num_of_layers; i++) {
		cudaMemcpyAsync(gpu_w[i], w[i], sizeof(float) * (s[i-1] * s[i]), cudaMemcpyHostToDevice,default_stream);
		cudaMemcpyAsync(gpu_b[i], b[i], sizeof(float) * (s[i]), cudaMemcpyHostToDevice,default_stream);
	}
	cudaStreamSynchronize(default_stream);

	// SECTION 4 :
	// train function - missing : update_sums(...) and gradient_descent(...) (check c++ code in the other file)
	struct timeval t1, t2;
	double time, time_c, time_h;
	gettimeofday(&t1, 0);
	train(num_of_layers, s, w, b, alfa, delta, sigm_derivative);
	gettimeofday(&t2, 0);
	time_h = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec)/ 1000.0;

	gettimeofday(&t1, 0);
	cuda_train(num_of_layers, s,gpu_w, b,gpu_alfa,gpu_delta,gpu_sigm_derivative);
	gettimeofday(&t2, 0);
	time_c = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec)/ 1000.0;
	printf("gpu time %0.6f , cpu time %0.6f \n",time_h,time_c);
	printf("Accelaration %0.6f %\n",((time_c-time_h)*2)/(time_h+time_c)*100);
	printf("success\n");
	return 0;
}

void cuda_train(int num_of_layers, int* s, float** w, float** b, float** alfa,
		float** delta, float** sigm_derivative) {
	// float learning_rate = 0.5;
	int epochs = 1;
	int batch_size = 1;
	int yd = 0;
	float* y, *cost;
	int blocks = 0;
	int numofthreads = 256;
	int multiplier;
	float cache = 11000 * sizeof(float);
	float* a = new float[s[0]];
	for (int ep = 0; ep < epochs; ep += (batch_size)) {
		// reset_sums(); --> NO CUDA VERSION OF IT
		for (int batch = 0; batch < batch_size; batch++) {
			// alfa[0] = read_tuple(ep + batch, &y_int); --> NO CUDA VERSION OF IT
			// since we don't read alfa[0] from file (in order to proper simulate it)
			// we will update alfa[0] with random values in each iteration
			// in any case, time would be wasted ,in order alfa[0] to be transfered in gpu
			for (int i = 0; i < s[0]; i++) {
				a[i] = getRandom(-1, 1);
			}
			// same goes for yd (y desired) READING VERSION FOR .CU FILE ISN'T YET CREATED
			yd = 0;
			y = transformOutput(yd, s[num_of_layers - 1]);

			// feedforward(&alfa[0]);
			cudaMemcpy(alfa[0], a, sizeof(float) * (s[0]), cudaMemcpyHostToDevice);
			for (int i = 1; i < num_of_layers; i++) {
				multiplier = floor(s[i - 1] / numofthreads) + 1;
				if (s[i-1] < numofthreads) {
					multiplier = 1;
				}
				feedforward<<<s[i], numofthreads, cache>>>(alfa[i],numofthreads, s[i], w[i], alfa[i-1], multiplier, s[i-1],b[i], sigm_derivative[i]);
				cudaDeviceSynchronize();
				// copy data back

			}

			// update_sums(); --> NO CUDA VERSION OF IT
		}
		// gradient_descent(learning_rate, batch_size); --> NO CUDA VERSION OF IT
	}

}

void train(int num_of_layers, int* s, float** w, float** b, float** alfa,
		float** delta, float** sigm_derivative) {
	// float learning_rate = 0.5;
	int epochs = 1;
	int batch_size = 1;
	int yd = 0;
	float* y, *cost;
	for (int ep = 0; ep < epochs; ep += (batch_size)) {
		// reset_sums(); --> NO CUDA VERSION OF IT
		for (int batch = 0; batch < batch_size; batch++) {
			// alfa[0] = read_tuple(ep + batch, &y_int); --> NO CUDA VERSION OF IT
			// since we don't read alfa[0] from file (in order to proper simulate it)
			// we will update alfa[0] with random values in each iteration
			// in any case, time would be wasted ,in order alfa[0] to be transfered in gpu
			for (int i = 0; i < s[0]; i++) {
				alfa[0][i] = getRandom(-1, 1);
			}
			// same goes for yd (y desired) READING VERSION FOR .CU FILE ISN'T YET CREATED
			yd = 0;
			y = transformOutput(yd, s[num_of_layers - 1]);

			// feedforward(&alfa[0]);
			for (int i = 1; i < num_of_layers; i++) {
				cpu_feedforward(alfa[i - 1], s[i - 1], s[i], &alfa[i], w[i],
						b[i], sigm_derivative[i]);
			}
			// NO TIME TO WRITE A CUDA IMPLEMENTATIION FOR THEM
			/*
			cost = cost_derivative(alfa[num_of_layers - 1], y,
					s[num_of_layers - 1]);

			delta[num_of_layers - 1] = hadamard_product(s[num_of_layers - 1],
					cost, sigm_derivative[num_of_layers - 1]);

			// backpropagate(delta[num_of_layers-1]);
			for (int i = num_of_layers - 2; i > 0; i--) {
				cpu_backpropagate(delta[i + 1], s[i], s[i + 1], &delta[i],
						sigm_derivative[i], w[i + 1]);
			}
			*/
			// update_sums(); --> NO CUDA VERSION OF IT
		}
		// gradient_descent(learning_rate, batch_size); --> NO CUDA VERSION OF IT
	}

}

float getRandom(int min, int max) {
	return (((max - min) * ((float) rand() / (float) RAND_MAX) + min) * 100)
			/ 100;
}

float* transformOutput(int output, int size) {
	// transforms a singleton input (named output:int) into
	// a vector (named result:*double)
	float* result = new float[size];
	for (int i = 0; i < size; i++) {
		result[i] = 0;
	}
	result[output] = 1;
	return result;
}

float* cost_derivative(float* a, float* y, int size) {
	// derivative of C with respect to a (a == output layer's content   )
	float* result = new float[size];
	for (int i = 0; i < size; i++) {
		result[i] = a[i] - y[i];
	}
	return result;
}

// FOR FEEDFORWARD IN CPU
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

void cpu_feedforward(float* a_old, int rows, int cols, float** a_new, float* w,
		float* b, float* sigm_der_result) {
	a_new[0] = compute_z(a_old, w, b, rows, cols);
	sigmoid(&a_new[0], cols);
	compute_sigm_der(a_new[0], sigm_der_result, cols);
}

// FOR BACKPROPAGATE
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

