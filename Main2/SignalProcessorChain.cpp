#include "SignalProcessorChain.h"

SignalProcessorChain::SignalProcessorChain(size_t datalen, SignalProcessor** chain, size_t chainlen) : SignalProcessor(datalen)
{

	this->chain = chain;
	this->chainlen = chainlen;
}

SignalProcessorChain::~SignalProcessorChain()
{
	for (size_t x = 0; x < this->chainlen; x++)
	{
		delete this->chain[x];
	}
	delete[] this->chain;
}

void SignalProcessorChain::reset()
{
	for (size_t x = 0; x < this->chainlen; x++)
	{
		this->chain[x]->reset();
	}
}

void SignalProcessorChain::process_buffer(SignalBuffer_t* buffer, size_t channel)
{
	for (size_t x = 0; x < this->chainlen; x++)
	{
		this->chain[x]->process_buffer(buffer, channel);
	}
}

int SignalProcessorChain::has_extra_samples()
{
	int has_extra = 0;
	for (size_t x = 0; !has_extra && x < this->chainlen; x++)
	{
		has_extra = chain[x]->has_extra_samples();
	}
	return has_extra;
}

void SignalProcessorChain::get_extra_samples(SignalBuffer_t* buffer, size_t channel)
{
	for (size_t i = 0; i < this->chainlen; i++)
	{
		SignalProcessor* top = this->chain[i];
		if (top->has_extra_samples())
		{
			top->get_extra_samples(buffer, channel);
			for (size_t j = i+1; j < this->chainlen; j++)
			{
				this->chain[j]->process_buffer(buffer, channel);
			}
			return;
		}
	}
	
	buffer->size = 0;
	return;
}