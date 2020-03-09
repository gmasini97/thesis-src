
#include "IFTT.h"

void ifft(SignalBuffer_t* buffer, size_t channel)
{
	size_t size = get_channel_buffer_size(*buffer);
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

IFFTProcessor::IFFTProcessor(size_t datalen) : SignalProcessor(datalen)
{
}
IFFTProcessor::~IFFTProcessor()
{
}

void IFFTProcessor::process_buffer(SignalBuffer_t* buffer, size_t channel)
{
	ifft(buffer, channel);
}