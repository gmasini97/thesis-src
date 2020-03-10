#include "DFT.h"

void dft(SignalBuffer_t* buffer, size_t channel)
{
	size_t size = get_channel_buffer_size(*buffer, channel);
	
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

DFTProcessor::DFTProcessor(AbstractSignalProcessor* p, BitMask channels_to_process) : SignalProcessor(p, channels_to_process)
{
}

DFTProcessor::~DFTProcessor()
{
}

void DFTProcessor::process_buffer(SignalBuffer_t* buffer)
{
	size_t channels = get_channels(*buffer);
	for (size_t c = 0 ; c < channels; c++) {
		size_t size = get_channel_buffer_size(*buffer, c);
		if (has_to_process_channel(c) && size > 0)
		{
			dft(buffer, c);
			//set_channel_buffer_size(*buffer, c, size / 2 + 1);
		}
	}

	if (has_next_processor())
		get_next_processor()->process_buffer(buffer);
}