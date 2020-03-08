#pragma once

#include <stdlib.h>

struct SignalBuffer_t
{
	float* real;
	float* imaginary;
	size_t channels;
	size_t size;
};

SignalBuffer_t create_signal_buffer(float* real, float* imaginary, size_t channels, size_t size);


class AbstractSignalProcessor
{
public:
	virtual size_t getDataLen() = 0;
	virtual void reset() = 0;
	virtual void process_buffer(SignalBuffer_t* buffer, size_t channel) = 0;
	virtual int has_extra_samples() = 0;
	virtual size_t get_extra_samples(float* real, float* imaginary, size_t buffer_size) = 0;
};


class SignalProcessor : public AbstractSignalProcessor
{
private:
	size_t datalen;
	size_t readcount;
protected:
	size_t get_read_count();
public:
	SignalProcessor(size_t datalen, size_t channel);
	~SignalProcessor();
	size_t getDataLen();
	virtual void reset();
	virtual void process_buffer(SignalBuffer_t* buffer, size_t channel);
	virtual int has_extra_samples();
	virtual size_t get_extra_samples(float* real, float* imaginary, size_t buffer_size);
};

class SignalProcessorChain : public SignalProcessor
{
private:
	SignalProcessor** chain;
	size_t chainlen;
public:
	SignalProcessorChain(size_t datalen, SignalProcessor** chain, size_t chainlen);
	~SignalProcessorChain();
	void reset();
	void process_buffer(float* real, float* imaginary, size_t readcount);
};

class MultichannelSignalProcessor : public SignalProcessor
{
private:
	size_t channels;
	SignalProcessor** processors;
	float* bufferRe;
	float* bufferIm;
public:
	MultichannelSignalProcessor(size_t datalen, size_t channels, SignalProcessor** processors);
	~MultichannelSignalProcessor();
	void reset();
	void process_buffer(float* real, float* imaginary, size_t readcount);
};

