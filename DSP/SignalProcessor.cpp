# include "SignalProcessor.h"

SignalProcessor::SignalProcessor(AbstractSignalProcessor* next, BitMask channels_to_process)
{
	this->next = next;
	this->channels_to_process = channels_to_process;
}

SignalProcessor::~SignalProcessor(){}

AbstractSignalProcessor* SignalProcessor::get_next_processor()
{
	return this->next;
}

int SignalProcessor::has_to_process_channel(size_t channel)
{
	return is_bit_set(this->channels_to_process, channel);
}

int SignalProcessor::has_next_processor()
{
	return this->next != NULL;
}

int SignalProcessor::init(size_t max_buffer_size, size_t channels)
{
	if (has_next_processor())
		return this->next->init(max_buffer_size, channels);
	return 1;
}

void SignalProcessor::process_buffer(SignalBuffer_t* buffer){}