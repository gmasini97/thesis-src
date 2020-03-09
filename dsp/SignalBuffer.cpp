#include "SignalBuffer.h"


SignalBuffer_t create_signal_buffer(size_t max_buffer_length, size_t channels)
{
	SignalBuffer_t buffer;
	buffer.channels = channels;
	buffer.max_size = max_buffer_length;
	buffer.samples = new cuComplex[max_buffer_length];
	buffer.channel_size = new size_t[channels];

	return buffer;
}

void delete_signal_buffer(SignalBuffer_t buffer)
{
	if (buffer.samples != NULL)
		delete[] buffer.samples;
	if (buffer.channel_size != NULL)
		delete[] buffer.channel_size;

	buffer.channels = 0;
	buffer.max_size = 0;
}

void clear_signal_buffer(SignalBuffer_t buffer)
{
	for (size_t c = 0; c < buffer.channels; c++)
	{
		buffer.channel_size[c] = 0;
	}
}

void clear_signal_buffer_deep(SignalBuffer_t buffer)
{
	for (size_t i = 0; i < buffer.max_size; i++)
	{
		buffer.samples[i] = make_cuComplex(0,0);
	}

	for (size_t c = 0; c < buffer.channels; c++)
	{
		buffer.channel_size[c] = 0;
	}
}

size_t signal_buffer_from_floats(SignalBuffer_t buffer, float* real, float* imaginary, size_t size)
{
	clear_signal_buffer(buffer);

	size_t to_read = (size < buffer.max_size) ? size : buffer.max_size;
	size_t channels = buffer.channels;
	size_t to_read_by_channel = to_read / channels;

	for (size_t i = 0; i < to_read_by_channel; i++)
	{
		for (size_t c = 0; c < channels; c++)
		{ 
			size_t index = i * channels + c;
			cuComplex sample = make_cuComplex(real ? real[index] : 0, imaginary ? imaginary[index] : 0);
			set_signal_buffer_sample(buffer, c, i, sample);
		}
	}

	size_t total_read = (to_read_by_channel * channels < to_read) ? to_read_by_channel * channels : to_read;
	return total_read;
}

size_t signal_buffer_to_floats(SignalBuffer_t buffer, float* real, float* imaginary)
{
	size_t max_channel_size = get_max_channel_buffer_size(buffer);
	size_t channels = buffer.channels;

	for (size_t i = 0; i < max_channel_size; i++)
	{
		for (size_t c = 0; c < channels; c++)
		{
			cuComplex sample;
			if (i < get_channel_buffer_size(buffer, c))
				sample = get_signal_buffer_sample(buffer, c, i);
			else
				sample = make_cuComplex(0,0);

			size_t index = i * channels + c;
			if (real)
				real[index] = sample.x;

			if (imaginary)
				imaginary[index] = sample.y;
		}
	}

	return max_channel_size * channels;
}
/*
size_t signal_buffer_to_floats(SignalBuffer_t buffer, BitMask channelMask, float* real, float* imaginary)
{
	size_t max_channel_buffer_size;
	size_t channels = get_channels(buffer);

	for (size_t channel = 0; channels < )

	size_t channels = buffer.channels;

	for (size_t i = 0; i < to_read; i++)
	{
		cuComplex sample = get_signal_buffer_sample(buffer, channel, i);
		real[i] = sample.x;
		imaginary[i] = sample.y;
	}

	return to_read;
}*/