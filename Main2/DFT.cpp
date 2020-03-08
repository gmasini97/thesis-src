#include "DFT.h"

void dft(SignalBuffer_t* buffer, size_t channel)
{
	size_t channels = buffer->channels;
	size_t size = get_channel_buffer_size(*buffer);
	
	cuComplex* tmp = new cuComplex[size];

	cuComplex sample, s;

	for (size_t k = 0; k < size; k++)
	{
		tmp[k] = make_cuFloatComplex(0,0);
		for (size_t i = 0; i < size; i++)
		{
			s = cuComplex_exp(-2 * M_PI * k * i / size);
			sample = get_signal_buffer_sample(*buffer, channel, i);

			tmp[k] = cuCaddf(tmp[k], cuCmulf(sample, s));
		}
	}

	for (size_t k = 0; k < size; k++)
	{
		set_signal_buffer_sample(*buffer, channel, k, tmp[k]);
	}

	delete[] tmp;
}

DFTProcessor::DFTProcessor(size_t datalen) : SignalProcessor(datalen)
{
}

DFTProcessor::~DFTProcessor()
{
}

void DFTProcessor::process_buffer(SignalBuffer_t* buffer, size_t channel)
{
	dft(buffer, channel);
}