#include "GainProcessor.h"

GainProcessor::GainProcessor(AbstractSignalProcessor* next, BitMask channels_to_process, float re_gain, float im_gain) : SignalProcessor(next, channels_to_process)
{
	this->gain = make_cuComplex(re_gain, im_gain);
}

void GainProcessor::process_buffer(SignalBuffer_t* buffer)
{
	size_t channels = get_channels(*buffer);
	for (size_t channel = 0; channel < channels; channel++)
	{
		if (has_to_process_channel(channel)) {
			size_t size = get_channel_buffer_size(*buffer, channel);
			for (size_t i = 0; i < size; i++)
			{
				cuComplex sample = get_signal_buffer_sample(*buffer, channel, i);
				set_signal_buffer_sample(*buffer, channel, i, cuCmulf(sample, this->gain));
			}
		}
	}

	if (has_next_processor())
		get_next_processor()->process_buffer(buffer);
}