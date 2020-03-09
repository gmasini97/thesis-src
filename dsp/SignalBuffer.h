#pragma once

#include <stdlib.h>
#include <cuComplex.h>
#include "BitMaskUtils.h"

struct SignalBuffer_t
{
	cuComplex* samples;
	size_t channels;
	size_t* channel_size;
	size_t max_size;
};
typedef struct SignalBuffer_t SignalBuffer_t;


SignalBuffer_t create_signal_buffer(size_t max_buffer_length, size_t channels);
size_t signal_buffer_from_floats(SignalBuffer_t buffer, float* real, float* imaginary, size_t size);
size_t signal_buffer_to_floats(SignalBuffer_t buffer, float* real, float* imaginary);
//size_t signal_buffer_to_floats(SignalBuffer_t buffer, size_t channel, float* real, float* imaginary);
void delete_signal_buffer(SignalBuffer_t buffer);
void clear_signal_buffer(SignalBuffer_t buffer);
void clear_signal_buffer_deep(SignalBuffer_t buffer);






__host__ __device__ static inline SignalBuffer_t empty_signal_buffer()
{
	return {NULL, 0, NULL, 0};
}

__host__ __device__ static inline size_t get_channels(SignalBuffer_t buffer)
{
	return buffer.channels;
}

__host__ __device__ static inline size_t get_channel_buffer_size(SignalBuffer_t buffer, size_t channel)
{
	size_t channel_buffer_size = buffer.channel_size[channel];
	return channel_buffer_size;
}

__host__ __device__ static inline size_t get_max_buffer_size(SignalBuffer_t buffer)
{
	return buffer.max_size;
}

__host__ __device__ static inline size_t get_max_channel_buffer_size(SignalBuffer_t buffer)
{
	size_t max_size = 0;
	for (size_t channel = 0; channel < buffer.channels; channel++)
	{
		size_t channel_size = get_channel_buffer_size(buffer, channel);
		max_size = channel_size > max_size ? channel_size : max_size;
	}
	return max_size;
}

__host__ __device__ static inline size_t get_max_possible_channel_buffer_size(SignalBuffer_t buffer, size_t channel)
{
	size_t max_size = get_max_buffer_size(buffer);
	size_t size = max_size / buffer.channels;
	return size;
}

__host__ __device__ static inline int set_channel_buffer_size(SignalBuffer_t buffer, size_t channel, size_t size)
{
	size_t max_size = get_max_possible_channel_buffer_size(buffer, channel);
	if (size > max_size) {
		return 0;
	}
	else
	{
		buffer.channel_size[channel] = size;
		return 1;
	}
}

__host__ __device__ static inline size_t get_signal_buffer_channel_sample_index(SignalBuffer_t buffer, size_t channel, size_t index)
{
	return index * buffer.channels + channel;
}

__host__ __device__ static inline cuComplex get_signal_buffer_sample(SignalBuffer_t buffer, size_t channel, size_t index)
{
	if (index >= get_channel_buffer_size(buffer, channel))
		return make_cuComplex(0,0);
	return buffer.samples[get_signal_buffer_channel_sample_index(buffer, channel, index)];
}

__host__ __device__ static inline int set_signal_buffer_sample(SignalBuffer_t buffer, size_t channel, size_t index, cuComplex value)
{
	size_t current_buffer_size = get_channel_buffer_size(buffer, channel);
	if (index >= current_buffer_size)
	{
		size_t new_current_buffer_size = index + 1;
		if (!set_channel_buffer_size(buffer, channel, new_current_buffer_size)) {
			// buffer overflow
			return 0;
		}
	}
	buffer.samples[get_signal_buffer_channel_sample_index(buffer, channel, index)] = value;

	return 1;
}