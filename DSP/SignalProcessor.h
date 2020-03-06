#pragma once

#include <stdlib.h>


class SignalProcessor
{
private:
	size_t datalen;
public:
	SignalProcessor(size_t datalen);
	~SignalProcessor();
	size_t getDataLen();
	virtual void reset() = 0;
	virtual void process_buffer(float* real, float* imaginary, size_t readcount) = 0;
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

class MultichannelSignalProcessor
{
private:
	size_t datalen;
	size_t channels;
	SignalProcessor** processors;
	float* bufferRe;
	float* bufferIm;
public:
	MultichannelSignalProcessor(size_t datalen, size_t channels, SignalProcessor** processors);
	~MultichannelSignalProcessor();
	size_t getDataLen();
	void reset();
	void process_buffer(float* real, float* imaginary, size_t readcount);
};

class NoProcessor : public SignalProcessor
{
public:
	NoProcessor(size_t datalen);
	~NoProcessor();
	void reset();
	void process_buffer(float* real, float* imaginary, size_t readcount);
};
