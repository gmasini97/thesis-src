#include "CUDAConvolver.cuh"

#define SIGNAL_CHANNEL 0

__device__ size_t cuda_bounded_index(SignalBuffer_t buffer, size_t channel, size_t index)
{
	size_t channel_size = get_channel_buffer_size(buffer, channel);
	return index >= channel_size ? index % channel_size : index;
}

size_t bounded_index(size_t max, size_t channel, size_t index)
{
	return index >= max ? index % max : index;
}

__global__ void cudaconvolver_kernel_set_size(SignalBuffer_t buf, size_t channel, size_t count)
{
	if (get_channel_buffer_size(buf, channel) < count)
		set_channel_buffer_size(buf, channel, count);
}

__global__ void cudaconvolver_kernel_output(SignalBuffer_t device_buffer, SignalBuffer_t signal, SignalBuffer_t tmp, size_t channel, size_t temp_index)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	size_t out_size = get_channel_buffer_size(tmp, channel);
	if (k >= out_size)
		return;

	size_t signal_size = get_channel_buffer_size(signal, SIGNAL_CHANNEL);

	cuComplex temp = make_cuComplex(0, 0);
	cuComplex signal_sample, input_sample;

	size_t index = cuda_bounded_index(tmp, channel, temp_index + k);
	cuComplex temp_sample = get_signal_buffer_sample(tmp, channel, index);

	for (int i = 0; i < signal_size; i++)
	{
		signal_sample = get_signal_buffer_sample(signal, SIGNAL_CHANNEL, i);
		if (i > k)
			input_sample = make_cuComplex(0,0);
		else
			input_sample = get_signal_buffer_sample(device_buffer, channel, k-i);
		temp_sample = cuCaddf(temp_sample, cuCmulf(signal_sample, input_sample));
	}
	//temp_sample = get_signal_buffer_sample(signal, channel, k);
	set_signal_buffer_sample(tmp, channel, index, temp_sample);
}

__global__ void cudaconvolver_kernel_copy(SignalBuffer_t device_buffer, SignalBuffer_t tmp, size_t channel, size_t temp_index)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	size_t tmp_size = get_channel_buffer_size(tmp, channel);
	size_t out_size = get_channel_buffer_size(device_buffer, channel);
	if (k >= tmp_size)
		return;

	size_t index = cuda_bounded_index(tmp, channel, temp_index + k);
	cuComplex sample = get_signal_buffer_sample(tmp, channel, index);
	set_signal_buffer_sample(device_buffer, channel, k, sample);
	if (k < out_size)
		set_signal_buffer_sample(tmp, channel, index, make_cuComplex(0,0));
}


CUDAConvolver::CUDAConvolver(AbstractSignalProcessor* next, BitMask channels_to_process, SignalBuffer_t signal) : CUDASignalProcessor(next, channels_to_process)
{
	this->signal = signal;
}

CUDAConvolver::~CUDAConvolver()
{
	delete_signal_buffer(this->signal);
	cuda_deallocate_signal_buffer(&(this->tmp));
	cuda_deallocate_signal_buffer(&(this->device_signal));
	delete[] this->temp_indexes;
	delete[] this->samples_remaining;
	CUDASignalProcessor::~CUDASignalProcessor();
}

int CUDAConvolver::init(size_t max_buffer_size, size_t channels)
{
	cudaError_t status;
	size_t extra_size = channels * get_max_buffer_size(this->signal) - 1;

	status = cuda_allocate_signal_buffer(&(this->tmp), max_buffer_size + extra_size, channels);
	if (check_cuda_status(status))
		return 0; // check this

	status = cuda_clear_signal_buffer_deep(this->tmp);
	if (check_cuda_status(status))
		return 0; // check this

	status = cuda_allocate_signal_buffer(&(this->device_signal), get_max_buffer_size(signal), get_channels(signal));
	if (check_cuda_status(status))
		return 0; // check this

	transfer_buffer_host_to_device(&(this->device_signal), this->signal);

	this->temp_indexes = new size_t[channels]{0};
	this->samples_remaining = new size_t[channels]{0};
	return CUDASignalProcessor::init(max_buffer_size, channels);
}

void CUDAConvolver::exec_kernel(SignalBuffer_t* host_buffer, SignalBuffer_t* device_buffer)
{
	LOG("CUDAConvolver kernel start\n");

	cudaError_t status;
	size_t channels = get_channels(*device_buffer);

	cudaStream_t* streams = new cudaStream_t[channels]{ NULL };

	for (size_t channel = 0; channel < channels; channel++)
	{
		if (!has_to_process_channel(channel))
			continue;

		size_t buffer_size = get_channel_buffer_size(*host_buffer, channel);
		size_t signal_size = get_channel_buffer_size(signal, SIGNAL_CHANNEL);
		size_t outcount = buffer_size + signal_size - 1;
		size_t remaining_samples = this->samples_remaining[channel];


		LOG("CUDAConvolver starting stream %lli\n", channel);

		status = cudaStreamCreate(streams + channel);
		check_cuda_status(status, "stream create");
		cudaStream_t stream = streams[channel];

		size_t temp_index = temp_indexes[channel];

		size_t bounded_max = get_max_possible_channel_buffer_size(tmp, channel);
		cudaconvolver_kernel_set_size << <1, 1, 0, stream >> > (this->tmp, channel, bounded_max);


		if ((buffer_size == 0 || signal_size == 0))
		{
			if (remaining_samples > 0){
				size_t max_size = get_max_possible_channel_buffer_size(*host_buffer, channel);
				size_t to_read = max_size < remaining_samples ? max_size : remaining_samples;
				size_t tmp_size = get_max_buffer_size(tmp);
				tmp_size = tmp_size < to_read + temp_index ? tmp_size : to_read+temp_index;

				dim3 threadsPerBlock;
				dim3 blocks;
				get_threads_blocks_count(to_read, threadsPerBlock, blocks);

				cudaconvolver_kernel_set_size << <1, 1, 0, stream >> > (*device_buffer, channel, to_read);
				cudaconvolver_kernel_copy << <blocks, threadsPerBlock, 0, stream >> > (*device_buffer, this->tmp, channel, temp_index);

				this->temp_indexes[channel] = bounded_index(bounded_max, channel, temp_index + to_read);
				this->samples_remaining[channel] = remaining_samples - to_read;
			}
			continue;
		}

		dim3 threadsPerBlock;
		dim3 blocks;
		get_threads_blocks_count(outcount, threadsPerBlock, blocks);

		cudaconvolver_kernel_output << <blocks, threadsPerBlock, 0, stream >> > (*device_buffer, this->device_signal, this->tmp, channel, temp_index);
		cudaconvolver_kernel_copy << <blocks, threadsPerBlock, 0, stream >> > (*device_buffer, this->tmp, channel, temp_index);

		LOG("CUDAConvolver started stream %lli\n", channel);

		this->temp_indexes[channel] = bounded_index(outcount, channel, temp_index + buffer_size);
		this->samples_remaining[channel] = signal_size - 1;
	}

	for (size_t channel = 0; channel < channels; channel++) {
		if (streams[channel] != NULL) {
			LOG("CUDAConvolver waiting stream %lli\n", channel);
			status = cudaStreamSynchronize(streams[channel]);
			check_cuda_status(status, "stream sync");
			status = cudaStreamDestroy(streams[channel]);
			check_cuda_status(status, "stream destr");
			LOG("CUDAConvolver destroyed stream %lli\n", channel);
		}
	}
}








CUDAConvolver* create_cuda_convolver_from_file(AbstractSignalProcessor* next, BitMask mask, std::string filename, size_t conv_size)
{
	SF_INFO info;
	memset(&info, 0, sizeof(SF_INFO));
	SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &info);

	if (info.channels != 1) {
		std::cout << "only 1 channel convolution kernel allowed" << std::endl;
		return NULL;
	}

	float* real = new float[conv_size];
	float* imag = new float[conv_size] {0};

	size_t actual_read = sf_read_float(file, real, conv_size);

	SignalBuffer_t buffer = create_signal_buffer(conv_size, 1);
	signal_buffer_from_floats(buffer, real, imag, actual_read);

	sf_close(file);
	delete[] real;
	delete[] imag;

	return new CUDAConvolver(next, mask, buffer);
}