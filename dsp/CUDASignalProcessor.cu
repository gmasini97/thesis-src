#include "CUDASignalProcessor.cuh"

#define MAX_THREADS_PER_BLOCK 1024

void get_threads_blocks_count(size_t processes, dim3 &threadsPerBlock, dim3 &blocks)
{
	size_t tpb = processes > MAX_THREADS_PER_BLOCK ? MAX_THREADS_PER_BLOCK : processes;
	size_t r = processes % tpb > 0 ? 1 : 0;
	size_t b = processes / tpb + r;
	b = b == 0 ? 1 : b;

	threadsPerBlock = dim3(tpb);
	blocks = dim3(b);
}

cudaError_t transfer_buffer_host_to_device(SignalBuffer_t* device, SignalBuffer_t host) {
	cudaError_t status;

	size_t size = get_max_buffer_size(*device);
	size_t channels = get_channels(*device);

	status = cudaMemcpy(device->samples, host.samples, size * sizeof(device->samples[0]), cudaMemcpyHostToDevice);
	status = cudaMemcpy(device->channel_size, host.channel_size, channels * sizeof(device->channel_size[0]), cudaMemcpyHostToDevice);

	return status;
}

cudaError_t transfer_buffer_device_to_host(SignalBuffer_t* host, SignalBuffer_t device) {
	cudaError_t status;

	size_t size = get_max_buffer_size(*host);
	size_t channels = get_channels(*host);

	status = cudaMemcpy(host->samples, device.samples, size * sizeof(host->samples[0]), cudaMemcpyDeviceToHost);
	status = cudaMemcpy(host->channel_size, device.channel_size, channels * sizeof(host->channel_size[0]), cudaMemcpyDeviceToHost);

	return status;
}

cudaError_t cuda_allocate_signal_buffer(SignalBuffer_t* buffer, size_t datalen, size_t channels)
{
	cudaError_t status;

	cuComplex* samples;
	status = cudaMalloc((void**) & (samples), datalen * sizeof(cuComplex));
	if (status != cudaSuccess) return status;

	size_t* channel_size;
	status = cudaMalloc((void**) & (channel_size), channels * sizeof(size_t));
	if (status != cudaSuccess) return status;

	buffer->samples = samples;
	buffer->max_size = datalen;
	buffer->channel_size = channel_size;
	buffer->channels = channels;

	return status;
}

cudaError_t cuda_clear_signal_buffer_deep(SignalBuffer_t buffer)
{
	size_t max_size = buffer.max_size;
	size_t channels = buffer.channels;

	cudaError_t status;
	
	cuComplex* tmp = new cuComplex[max_size];
	for (size_t i = 0 ; i < max_size; i++)
		tmp[i] = make_cuComplex(0,0);
	status = cudaMemcpy(buffer.samples, tmp, max_size * sizeof(cuComplex), cudaMemcpyHostToDevice);
	delete[] tmp;
	if (status != cudaSuccess)
		return status;

	size_t* zero_channels = new size_t[channels]{0};
	status = cudaMemcpy(buffer.channel_size, zero_channels, channels * sizeof(size_t), cudaMemcpyHostToDevice);
	delete[] zero_channels;
	if (status != cudaSuccess)
		return status;

	return cudaSuccess;
}

void cuda_deallocate_signal_buffer(SignalBuffer_t* buffer)
{
	cudaFree(buffer->samples);
	cudaFree(buffer->channel_size);
	buffer->channels = 0;
	buffer->max_size = 0;
}

CUDASignalProcessor::CUDASignalProcessor(AbstractSignalProcessor* previous, BitMask channels_to_process) : SignalProcessor(previous, channels_to_process)
{
	this->err = 0;
	this->time = 0;
	this->parallel_channels = 0;
}

CUDASignalProcessor::~CUDASignalProcessor()
{
	cuda_deallocate_signal_buffer(&(this->device_buffer));
	SignalProcessor::~SignalProcessor();
}

int CUDASignalProcessor::init(size_t max_buffer_size, size_t channels)
{
	cudaError_t status;

	status = cudaSetDevice(0);
	if (check_cuda_status(status, "set_device")) return 0;

	status = cuda_allocate_signal_buffer(&(this->device_buffer), max_buffer_size, channels);
	if (check_cuda_status(status, "alloc_dev_buffer")) return 0;

	return SignalProcessor::init(max_buffer_size, channels);
}

int CUDASignalProcessor::check_cuda_status(cudaError_t status, const char* msg)
{
	if (status != cudaSuccess)
	{
		this->err = 1;
		std::cout << "[" << msg << "] cuda err: " << cudaGetErrorString(status) << std::endl;
	}
	return this->err;
}

void CUDASignalProcessor::process_buffer(SignalBuffer_t* buffer)
{
	if (has_previous_processor())
		get_previous_processor()->process_buffer(buffer);
	LOG("CUDASignalProcessor process_buffer start\n");
	cudaError_t status;
	size_t channels = get_channels(*buffer);
	auto start = std::chrono::high_resolution_clock::now();

	status = transfer_buffer_host_to_device(&(this->device_buffer), *buffer);
	if (check_cuda_status(status, "buffer h>d")) return;

	status = cudaGetLastError();
	if (check_cuda_status(status, "last_err")) return;

	status = cudaDeviceSynchronize();
	if (check_cuda_status(status, "dev_synch")) return;

	this->exec_kernel(buffer, &(this->device_buffer));

	status = cudaGetLastError();
	if (check_cuda_status(status, "last_err")) return;

	status = cudaDeviceSynchronize();
	if (check_cuda_status(status, "dev_synch")) return;

	status = transfer_buffer_device_to_host(buffer, this->device_buffer);
	if (check_cuda_status(status, "buffer d>h")) return;

	auto end = std::chrono::high_resolution_clock::now();

	this->time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

}

int CUDASignalProcessor::get_err()
{
	return this->err;
}

long long int CUDASignalProcessor::get_time()
{
	return this->time;
}