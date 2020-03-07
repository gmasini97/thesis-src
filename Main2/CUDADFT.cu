#include "CUDADFT.h"

#define PI 3.141592654f

__global__ void cudadft_kernel_dft(float* real, float* imaginary, float* rt, float* it, int readcount)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	
	float sr, si;
	float re = 0;
	float im = 0;

	for (int i = 0; i < readcount; i++)
	{
		sr = cos(2.0f * PI * k * i / readcount);
		si = -sin(2.0f * PI * k * i / readcount);

		re += real[i] * sr - imaginary[i] * si;
		im += real[i] * si + imaginary[i] * sr;
	}

	rt[k] = re;
	it[k] = im;
}
__global__ void cudadft_kernel_copy(float* real, float* imaginary, float* rt, float* it, int readcount)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	real[k] = rt[k];
	imaginary[k] = it[k];
}


CUDADFT::CUDADFT(size_t datalen) : CUDASignalProcessor(datalen)
{
	cudaError_t status;

	status = cudaMalloc((void**) & (this->reTmp), datalen * sizeof(float));
	if (check_cuda_status(status)) goto fin;

	status = cudaMalloc((void**) & (this->imTmp), datalen * sizeof(float));
	if (check_cuda_status(status)) goto fin;
fin:
}

CUDADFT::~CUDADFT()
{
	cudaFree(this->reTmp);
	cudaFree(this->imTmp);
}

void CUDADFT::exec_kernel(float* real, float* imaginary, size_t readcount)
{
	cudaError_t status;

	dim3 threadsPerBlock(128);
	dim3 blocks(readcount / threadsPerBlock.x);
	cudadft_kernel_dft <<<blocks, threadsPerBlock>>> (real, imaginary, reTmp, imTmp, readcount);

	status = cudaGetLastError();
	if (check_cuda_status(status)) return;

	status = cudaDeviceSynchronize();
	if (check_cuda_status(status)) return;

	cudadft_kernel_copy <<<blocks, threadsPerBlock >>> (real, imaginary, reTmp, imTmp, readcount);
}