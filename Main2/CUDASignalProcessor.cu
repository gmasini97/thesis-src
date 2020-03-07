#include "CUDASignalProcessor.h"

CUDASignalProcessor::CUDASignalProcessor(size_t datalen) : SignalProcessor(datalen)
{
	cudaError_t status;

	this->err = 0;
	this->time = 0;

	status = cudaSetDevice(0);
	if (check_cuda_status(status)) goto fin;

	status = cudaMalloc((void**) &(this->re), datalen * sizeof(float));
	if (check_cuda_status(status)) goto fin;

	status = cudaMalloc((void**) &(this->im), datalen * sizeof(float));
	if (check_cuda_status(status)) goto fin;

fin:
}

CUDASignalProcessor::~CUDASignalProcessor()
{
	cudaFree(this->re);
	cudaFree(this->im);
}

int CUDASignalProcessor::check_cuda_status(cudaError_t status)
{
	if (status != cudaSuccess)
	{
		this->err = 1;
		std::cout << "cuda err: " << cudaGetErrorString(status) << std::endl;
	}
	return this->err;
}

void CUDASignalProcessor::process_buffer(float* real, float* imaginary, size_t readcount)
{
	cudaError_t status;

	auto start = std::chrono::high_resolution_clock::now();

	status = cudaMemcpy(this->re, real, readcount * sizeof(float), cudaMemcpyHostToDevice);
	if (check_cuda_status(status)) return;

	status = cudaMemcpy(this->im, imaginary, readcount * sizeof(float), cudaMemcpyHostToDevice);
	if (check_cuda_status(status)) return;

	this->exec_kernel(this->re, this->im, readcount);

	status = cudaGetLastError();
	if (check_cuda_status(status)) return;

	status = cudaDeviceSynchronize();
	if (check_cuda_status(status)) return;

	status = cudaMemcpy(real, this->re, readcount * sizeof(float), cudaMemcpyDeviceToHost);
	if (check_cuda_status(status)) return;

	status = cudaMemcpy(imaginary, this->im, readcount * sizeof(float), cudaMemcpyDeviceToHost);
	if (check_cuda_status(status)) return;

	auto end = std::chrono::high_resolution_clock::now();
	
	this->time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

}

void CUDASignalProcessor::reset()
{
	this->time = 0;
}

int CUDASignalProcessor::get_err()
{
	return this->err;
}

long long int CUDASignalProcessor::get_time()
{
	return this->time;
}