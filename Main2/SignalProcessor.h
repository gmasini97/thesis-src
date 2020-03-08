#pragma once

#include <stdlib.h>
#include <cuComplex.h>

struct SignalBuffer_t
{
	cuComplex* samples;
	size_t channels;
	size_t size;
	size_t maxsize;
};

SignalBuffer_t allocate_signal_buffer(size_t maxsize);
void reallocate_signal_buffer(SignalBuffer_t* buffer, size_t maxsize);
void signal_buffer_from_floats(SignalBuffer_t* buffer, float* real, float* imaginary, size_t channels, size_t size);
void signal_buffer_to_floats(SignalBuffer_t buffer, float* real, float* imaginary, size_t channel);
void deallocate_signal_buffer(SignalBuffer_t* buffer);



class AbstractSignalProcessor
{
public:
	virtual size_t getDataLen() = 0;
	virtual void reset() = 0;
	virtual void process_buffer(SignalBuffer_t* buffer, size_t channel) = 0;
	virtual int has_extra_samples() = 0;
	virtual void get_extra_samples(SignalBuffer_t* buffer, size_t channel) = 0;
};


class SignalProcessor : public AbstractSignalProcessor
{
private:
	size_t datalen;
public:
	SignalProcessor(size_t datalen);
	~SignalProcessor();
	size_t getDataLen();
	virtual void reset();
	virtual void process_buffer(SignalBuffer_t* buffer, size_t channel);
	virtual int has_extra_samples();
	virtual void get_extra_samples(SignalBuffer_t* buffer, size_t channel);
};




__host__ __device__ static inline size_t get_channel_buffer_size(SignalBuffer_t buffer)
{
	return buffer.size / buffer.channels;
}

__host__ __device__ static inline size_t get_max_channel_buffer_size(SignalBuffer_t buffer)
{
	return buffer.maxsize / buffer.channels;
}

__host__ __device__ static inline void set_channel_buffer_size(SignalBuffer_t* buffer, size_t size)
{
	buffer->size = size * buffer->channels;
}

__host__ __device__ static inline cuComplex get_signal_buffer_sample(SignalBuffer_t buffer, size_t channel, size_t index)
{
	return buffer.samples[index * buffer.channels + channel];
}

__host__ __device__ static inline void set_signal_buffer_sample(SignalBuffer_t buffer, size_t channel, size_t index, cuComplex value)
{
	buffer.samples[index * buffer.channels + channel] = value;
}