#include <cuda_runtime.h>
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization
#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <filesystem>

#define DIMX 32*8;

using namespace std;


/*Debugging purpose */
float* gpu_to_cpu(float* in, int nbytes) {
	float* out = (float*)malloc(nbytes);
	checkCudaErrors(cudaMemcpy(out, in, nbytes, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());

	return out;
}

//Code are modified based on https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
void ReadMNIST(char* img_file, char* label_file, int NumberOfImages, int DataOfAnImage, float* arr, int* arr_lbl)
{
	//arr.resize(NumberOfImages, vector<float>(DataOfAnImage));
	//arr_lbl.resize(NumberOfImages);

	ifstream file(img_file, ios::binary);
	ifstream file_lbl(label_file, ios::binary);
	if (file.is_open() && file_lbl.is_open())
	{
		printf("file is open \n");
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		file_lbl.read((char*)&magic_number, sizeof(magic_number));
		file_lbl.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for (int i = 0; i < NumberOfImages; ++i)
		{
			unsigned char temp = 0;
			file_lbl.read((char*)&temp, sizeof(temp));
			arr_lbl[i] = (int)temp;

			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));

					arr[i*(n_rows*n_cols) + (n_rows*r) + c] = ((float)temp) / 255;
				}
			}
		}
	}
	file.close();
	file_lbl.close();

}

bool* one_hot_encode(int* lbl, int size) {
	bool* one = (bool *)malloc(size * 10 * sizeof(bool));
	for (int i = 0; i < size; i++) {
		for (int c = 0; c < 10; c++) {
			one[i * 10 + c] = (bool)0;
		}
		one[i * 10 + lbl[i]] = (bool)1;
	}
	return one;
}

__global__ void add_bias(float* added, float* ori, int nrows, int ncols) {
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	if (ix < nrows) {
		for (int i = 0; i < ncols; i++) {
			added[ix*(ncols + 1) + i] = ori[ix*ncols + i];
		}
		added[ix*(ncols + 1) + ncols] = (float) 1.0;
	}
}

float* append_bias(float* in, int rows, int cols) {
	float* temp;
	checkCudaErrors(cudaMalloc((float**)&temp, rows * (cols+1) * sizeof(float)));
	//checkCudaErrors(cudaMemset(temp, 0, rows * (cols + 1) * sizeof(float)));
	//checkCudaErrors(cudaDeviceSynchronize());

	int dimx = DIMX;
	dim3 block(dimx, 1);
	dim3 grid((rows + block.x - 1) / block.x, 1);

	add_bias << < grid , block >> > (temp, in, rows, cols);
	checkCudaErrors(cudaDeviceSynchronize());

	return temp;
}

// out = AB
// A:mxn; B:nxk, out:mxk
__global__ void mat_mul( float* out, float* A, float* B, int m, int n, int k) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	if (ix < m*k) {
		int row_id = ix / k;
		int col_id = ix % k;

		float tmp = 0;
		for (int i = 0; i < n; i++) {
			tmp += A[row_id*n + i] * B[i*k + col_id];
		}

		out[ix] = tmp;
	}
}

// softmax
__global__ void softmax( float* in, int nrows, int ncols ) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;

	if (ix < nrows) {
		double sum = 0;
		for (int i = 0; i < ncols; i++) {
			double tmp = exp( (double) in[ix*ncols + i]);
			sum += tmp;
		}

		for (int i = 0; i < ncols; i++) {
			double tmp = exp((double)in[ix*ncols + i]) / sum;
			in[ix*ncols + i] = (float) tmp;

			if (isnan(in[ix*ncols + i])) {
				printf("NAN: %f %f\n", in[ix*ncols + i], sum);
			}

		}

	}

}

float* forward(float* X, float* w, int num, int img_dim, int dimx_) {
	float* pred;
	checkCudaErrors(cudaMalloc((float**)&pred, num * 10 * sizeof(float)));

	int dimx = dimx_;
	dim3 block(dimx, 1);
	dim3 grid((num*10 + block.x - 1) / block.x, 1);
	mat_mul << <grid, block>> > (pred, X, w, num, img_dim+1, 10);
	//checkCudaErrors(cudaDeviceSynchronize());

	softmax << <grid, block>> > (pred, num, 10);
	//checkCudaErrors(cudaDeviceSynchronize());

	return pred;
}

// A = A-B
__global__ void mat_substract(float* A, float* B, int total) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	if (ix < total) {
		A[ix] = A[ix] - B[ix];
	}
}

__global__ void mat_substract_bool(float* A, bool* B, int total) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	if (ix < total) {
		A[ix] = A[ix] - ((float) B[ix]);
	}
}

// elemen-wise multi A = gamma * A
__global__ void element_wise(float* A, float gamma, int total) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	if (ix < total) {
		A[ix] = A[ix] * gamma;
	}
}

// out = A^T B
// A:nxm; B:nxk, out:mxk
__global__ void mat_mul_T(float* out, float* A, float* B, int m, int n, int k) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	if (ix < m*k) {
		int row_id = ix / k;
		int col_id = ix % k;

		float tmp = 0;
		for (int i = 0; i < n; i++) {
			tmp += A[i*m + row_id] * B[i*k + col_id];
		}

		out[ix] = tmp;
	}
}

void update(float* w, float* pred, bool* T, float* X, int num, int img_dim, float lr, int dimx_) {
	int dimx = dimx_;
	dim3 block(dimx, 1);
	dim3 grid((num * 10 + block.x - 1) / block.x, 1);
	mat_substract_bool << < grid, block >> > (pred, T, num*10);
	checkCudaErrors(cudaDeviceSynchronize());

	float* temp;
	checkCudaErrors(cudaMalloc((float**)&temp, num * (img_dim+1) * sizeof(float)));

	dim3 block_0(dimx, 1);
	dim3 grid_0((num * (img_dim+1) + block.x - 1) / block.x, 1);
	mat_mul_T << <grid_0, block_0>> > (temp, X, pred, img_dim+1, num, 10);
	//checkCudaErrors(cudaDeviceSynchronize());

	element_wise << <grid_0, block_0>> > (temp, lr, (img_dim+1)*10 );
	//checkCudaErrors(cudaDeviceSynchronize());

	mat_substract << < grid_0, block_0 >> > (w, temp, (img_dim + 1)*10);
	//checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(temp));
}

//out = argmax(A) == argmax(B)
__global__ void equal(bool* out, float* A, bool* B, int rows) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (ix < rows) {
		float max_A = 0;
		int max_A_idx = 0;

		int max_B_idx = 0;
		for (int i = 0; i < 10; i++) {
			if (A[ix*10 + i] > max_A) {
				max_A = A[ix * 10 + i];
				max_A_idx = i;
			}
			if (B[ix * 10 + i] == (bool) 1) {
				max_B_idx = i;
			}
		}
		
		if (max_A_idx == max_B_idx) {
			out[ix] = (bool)1;
		}
		else {
			out[ix] = (bool)0;
		}

	}
}

float get_error(float* pred, bool* T, int num, int dimx_) {
	bool* temp;
	checkCudaErrors(cudaMalloc((float**)&temp, num * sizeof(bool)));

	int dimx = dimx_;
	dim3 block(dimx, 1);
	dim3 grid((num + block.x - 1) / block.x, 1);
	equal << < grid, block >> > (temp, pred, T, num);
	//checkCudaErrors(cudaDeviceSynchronize());

	bool* temp_cpu = (bool*)malloc(num * sizeof(bool));
	checkCudaErrors(cudaMemcpy(temp_cpu, temp, num*sizeof(bool), cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(temp));

	float acc = 0;
	for (int i = 0; i < num; i++) {
		acc += ((float)temp_cpu[i]);
		//printf("%f ", ((float)temp_cpu[i]));
	}

	acc = acc / (float) num;

	free(temp_cpu);

	return acc;
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

	int train_num = 60000;
	int test_num = 10000;
	int img_dim = 28 * 28;
	int class_num = 10;
	float lr = 0.00001;
	int num_epochs = 1000;

	// Get training data
	char* train_images_path = "../../../training set/train-images.idx3-ubyte";
	char* train_lbl_path = "../../../training set/train-labels.idx1-ubyte";
	float* X_train = (float *)malloc(train_num * img_dim * sizeof(float));
	int* T_train = (int *)malloc(train_num * sizeof(int));

	ReadMNIST(train_images_path, train_lbl_path, train_num, img_dim, X_train, T_train);

	//Get Testing data
	char* test_images_path = "../../../test set/t10k-images.idx3-ubyte";
	char* test_lbl_path = "../../../test set/t10k-labels.idx1-ubyte";
	float* X_test = (float *)malloc(test_num * img_dim * sizeof(float));;
	int* T_test = (int *)malloc(test_num * sizeof(int));

	ReadMNIST(test_images_path, test_lbl_path, test_num, img_dim, X_test, T_test);

	//one hot encode labels
	bool* T_train_one = one_hot_encode(T_train, train_num);
	bool* T_test_one = one_hot_encode(T_test, test_num);

	//allocate device memory for image and labels
	float *d_X_train;
	bool *d_T_train_one;
	checkCudaErrors(cudaMalloc((float**)&d_X_train, train_num * img_dim * sizeof(float)));
	checkCudaErrors(cudaMalloc((bool**)&d_T_train_one, train_num * 10 * sizeof(bool)));

	float *d_X_test;
	bool *d_T_test_one;
	checkCudaErrors(cudaMalloc((float**)&d_X_test, test_num * img_dim * sizeof(float)));
	checkCudaErrors(cudaMalloc((bool**)&d_T_test_one, test_num * 10 * sizeof(bool)));

	//copy train and test data to device 
	checkCudaErrors(cudaMemcpy(d_X_train, X_train, train_num * img_dim * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_T_train_one, T_train_one, train_num * 10 * sizeof(bool), cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMemcpy(d_X_test, X_test, test_num * img_dim * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_T_test_one, T_test_one, test_num * 10 * sizeof(bool), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaDeviceSynchronize());
	free(T_train_one);
	free(T_test_one);
	free(X_train);
	free(X_test);

	//weight random init
	float* w;
	checkCudaErrors(cudaMalloc((float**)&w, (img_dim + 1) * class_num * sizeof(float)));
	srand(3);
	float* temp = (float *)malloc((img_dim + 1) * class_num * sizeof(float));
	for (int i = 0; i < (img_dim + 1) * class_num; i++) {
		temp[i] = (float)rand() / RAND_MAX;
	}

	checkCudaErrors(cudaMemcpy(w, temp, (img_dim+1) * class_num * sizeof(float), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaDeviceSynchronize());
	free(temp);

	//Append Bias to X_train and X_test
	float* d_X_train_bias = append_bias(d_X_train, train_num, img_dim);
	float* d_X_test_bias = append_bias(d_X_test, test_num, img_dim);
	//checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(d_X_train));
	checkCudaErrors(cudaFree(d_X_test));

	for (int warp = 14; warp <= 14; warp++) {
		int dimx = 32*warp; 

		auto iStart = chrono::high_resolution_clock::now();

		/*
		float* w_cpu = gpu_to_cpu(w, (img_dim + 1) * 10 * sizeof(float));
		for (int j = 0; j < 785; j++) {
			for (int i = 0; i < 10; i++) {
				printf("%f ", w_cpu[j*10 + i]);
			}
			//printf("\n");
		}


		printf("\n");

		free(w_cpu);
		*/

		for (int ep = 0; ep < num_epochs; ep++) {

			float* pred = forward(d_X_train_bias, w, train_num, img_dim, dimx);

			update(w, pred, d_T_train_one, d_X_train_bias, train_num, img_dim, lr, dimx);

			checkCudaErrors(cudaFree(pred));


			if (ep % 100 == 0 || ep == num_epochs - 1) {
				pred = forward(d_X_test_bias, w, test_num, img_dim, dimx);
				float acc_test = get_error(pred, d_T_test_one, test_num, dimx);
				checkCudaErrors(cudaFree(pred));

				pred = forward(d_X_train_bias, w, train_num, img_dim, dimx);
				float acc = get_error(pred, d_T_train_one, train_num, dimx);
				checkCudaErrors(cudaFree(pred));


				printf("Epoch: %d, Train Acc: %f, Test Acc: %f \n", ep, acc, acc_test);
			}
		}

		auto iStop = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::seconds>(iStop - iStart);

		// To get the value of duration use the count()
		// member function on the duration object
		printf("DIMX: %d \n", dimx);

		printf("Done ! Total Training time: %f sec. \n", (float)duration.count());
	}

	//free(X_train);
	free(T_train);
	//free(T_train_one);
	//free(X_test);
	free(T_test);
	//free(T_test_one);

	checkCudaErrors(cudaFree(d_X_train_bias));
	checkCudaErrors(cudaFree(d_X_test_bias));
	checkCudaErrors(cudaFree(d_T_train_one));
	checkCudaErrors(cudaFree(d_T_test_one));

	checkCudaErrors(cudaFree(w));


    return (0);
}
