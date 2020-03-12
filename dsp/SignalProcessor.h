#pragma once

#include "SignalBuffer.h"
#include "BitMaskUtils.h"
#include "LogUtils.h"

class AbstractSignalProcessor
{
public:
	virtual int has_to_process_channel(size_t channel) = 0;
	virtual AbstractSignalProcessor* get_next_processor() = 0;
	virtual int init(size_t max_buffer_size, size_t channels) = 0;
	virtual void process_buffer(SignalBuffer_t* buffer) = 0;
};


class SignalProcessor : public AbstractSignalProcessor
{
private:
	AbstractSignalProcessor* next;
	BitMask channels_to_process;
public:
	SignalProcessor(AbstractSignalProcessor* next, BitMask channels_to_process);
	~SignalProcessor();

	AbstractSignalProcessor* get_next_processor();
	int has_to_process_channel(size_t channel);
	int has_next_processor();

	virtual int init(size_t max_buffer_size, size_t channels);
	virtual void process_buffer(SignalBuffer_t* buffer);
};