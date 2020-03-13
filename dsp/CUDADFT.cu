#include "CUDADFT.cuh"

#define PI 3.141592654f

__global__ void cudadft_kernel_dft(SignalBuffer_t device_buffer, SignalBuffer_t tmp, size_t channel)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	size_t size = get_channel_buffer_size(device_buffer, channel);
	cuComplex temp = make_cuComplex(0, 0);
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


CUDADFT::CUDADFT(AbstractSignalProcessor* previous, BitMask channels_to_process) : CUDASignalProcessor(previous, channels_to_process)
{
}

CUDADFT::~CUDADFT()
{
	cuda_deallocate_signal_buffer(&(this->tmp));
	CUDASignalProcessor::~CUDASignalProcessor();
}

int CUDADFT::init(size_t max_buffer_size, size_t channels)
{
	cudaError_t status;
	status = cuda_allocate_signal_buffer(&(this->tmp), max_buffer_size, channels);
	if (check_cuda_status(status))
		return 0; // check this
	return CUDASignalProcessor::init(max_buffer_size, channels);
}

void CUDADFT::exec_kernel(SignalBuffer_t* host_buffer, SignalBuffer_t* device_buffer)
{
	LOG("CUDADFT kernel start\n");

	cudaError_t status;
	size_t channels = get_channels(*device_buffer);

	cudaStream_t* streams = new cudaStream_t[channels]{NULL};

	for (size_t channel = 0; channel < channels; channel++)
	{
		if (!has_to_process_channel(channel))
			continue;

		size_t readcount = get_channel_buffer_size(*host_buffer, channel);

		if (readcount <= 0)
			continue;

		size_t threads = readcount < 512 ? readcount : 512;

		dim3 threadsPerBlock(threads);
		dim3 blocks(readcount / threadsPerBlock.x);

		LOG("CUDADFT starting stream %lli\n", channel);

		status = cudaStreamCreate(streams+channel);
		check_cuda_status(status, "stream create");
		cudaStream_t stream = streams[channel];

		status = cudaMemcpy(tmp.channel_size + channel, device_buffer->channel_size + channel, sizeof(tmp.channel_size[0]), cudaMemcpyDeviceToDevice);

		cudadft_kernel_dft << <blocks, threadsPerBlock, 0, stream >> > (*device_buffer, this->tmp, channel);
		cudadft_kernel_copy << <blocks, threadsPerBlock, 0, stream >> > (*device_buffer, this->tmp, channel);

		LOG("CUDADFT started stream %lli\n", channel);

	}

	for (size_t channel = 0; channel < channels; channel++) {
		if (streams[channel] != NULL){
			LOG("CUDADFT waiting stream %lli\n", channel);
			status = cudaStreamSynchronize(streams[channel]);
			check_cuda_status(status, "stream sync");
			status = cudaStreamDestroy(streams[channel]);
			check_cuda_status(status, "stream destr");
			LOG("CUDADFT destroyed stream %lli\n", channel);
		}
	}
}