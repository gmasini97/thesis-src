#include "CUDADFT.cuh"

#define PI 3.141592654f

__global__ void cudadft_kernel_dft(SignalBuffer_t device_buffer, SignalBuffer_t tmp, size_t channel)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	size_t size = get_channel_buffer_size(device_buffer);
	cuComplex temp = make_cuComplex(0,0);
	cuComplex sample, s;

	for (int i = 0; i < size; i++)
	{
		sample = get_signal_buffer_sample(device_buffer, channel, i);

		s = cuComplex_exp(-2.0f * PI * k * i / size);

		temp = cuCaddf(temp, cuCmulf(sample, s));
	}

	set_signal_buffer_sample(tmp, channel, k, temp);
}

__global__ void cudadft_kernel_copy(SignalBuffer_t device_buffer, SignalBuffer_t tmp, size_t channel)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	cuComplex sample = get_signal_buffer_sample(tmp, channel, k);
	set_signal_buffer_sample(device_buffer, channel, k, sample);
}


CUDADFT::CUDADFT(size_t datalen) : CUDASignalProcessor(datalen)
{
	cudaError_t status;

	status = cuda_allocate_signal_buffer(&(this->tmp), datalen);
	if (check_cuda_status(status)) goto fin;

fin:
}

CUDADFT::~CUDADFT()
{
	cuda_deallocate_signal_buffer(&(this->tmp));
}

void CUDADFT::exec_kernel(SignalBuffer_t* host_buffer, SignalBuffer_t* device_buffer, size_t channel)
{
	cudaError_t status;

	this->tmp.channels = device_buffer->channels;
	this->tmp.size = device_buffer->size;

	size_t readcount = get_channel_buffer_size(*host_buffer);

	dim3 threadsPerBlock(128);
	dim3 blocks(readcount / threadsPerBlock.x);
	cudadft_kernel_dft <<<blocks, threadsPerBlock>>> (*device_buffer, this->tmp, channel);

	status = cudaGetLastError();
	if (check_cuda_status(status, "last_err_dft")) return;

	status = cudaDeviceSynchronize();
	if (check_cuda_status(status, "dev_synch_dft")) return;

	cudadft_kernel_copy <<<blocks, threadsPerBlock >>> (*device_buffer, this->tmp, channel);

	status = cudaGetLastError();
	if (check_cuda_status(status, "last_err2_dft")) return;

	status = cudaDeviceSynchronize();
	if (check_cuda_status(status, "dev_synch2_dft")) return;
}