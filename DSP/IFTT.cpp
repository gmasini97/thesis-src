
#include "IFTT.h"

void ifft(SignalBuffer_t* buffer, size_t channel)
{
	size_t size = get_channel_buffer_size(*buffer, channel);
	cuComplex sample;

	for (size_t k = 0; k < size; k++)
	{
		sample = get_signal_buffer_sample(*buffer, channel, k);
		sample = cuConjf(sample);
		set_signal_buffer_sample(*buffer, channel, k, sample);
	}

	fft(buffer, channel);

	cuComplex size_cmplx = make_cuComplex(size, 0);

	for (size_t i = 0; i < size; i++)
	{
		sample = get_signal_buffer_sample(*buffer, channel, i);
		sample = cuConjf(cuCdivf(sample, size_cmplx));
		set_signal_buffer_sample(*buffer, channel, i, sample);
	}
}

IFFTProcessor::IFFTProcessor(AbstractSignalProcessor* next, BitMask channels_to_process) : SignalProcessor(next, channels_to_process)
{
}

void IFFTProcessor::process_buffer(SignalBuffer_t* buffer)
{
	size_t channels = get_channels(*buffer);
	for (size_t channel = 0; channel < channels; channel++)
	{
		size_t size = get_channel_buffer_size(*buffer, channel);
		if (has_to_process_channel(channel) && size > 0) {
			ifft(buffer, channel);
			//set_channel_buffer_size(*buffer, channel, size * 2 - 1);
		}
	}

	if (has_next_processor())
		get_next_processor()->process_buffer(buffer);
}