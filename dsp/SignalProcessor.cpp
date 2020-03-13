# include "SignalProcessor.h"

SignalProcessor::SignalProcessor(AbstractSignalProcessor* previous, BitMask channels_to_process)
{
	this->previous = previous;
	this->channels_to_process = channels_to_process;
}

SignalProcessor::~SignalProcessor(){}

AbstractSignalProcessor* SignalProcessor::get_previous_processor()
{
	return this->previous;
}

int SignalProcessor::has_to_process_channel(size_t channel)
{
	return is_bit_set(this->channels_to_process, channel);
}

int SignalProcessor::has_previous_processor()
{
	return this->previous != NULL;
}

int SignalProcessor::init(size_t max_buffer_size, size_t channels)
{
	if (has_previous_processor())
		return this->previous->init(max_buffer_size, channels);
	return 1;
}

void SignalProcessor::process_buffer(SignalBuffer_t* buffer){}