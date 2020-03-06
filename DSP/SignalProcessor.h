#pragma once

#include <stdlib.h>


class SignalProcessor
{
private:
	size_t datalen;
public:
	SignalProcessor(size_t datalen);
	size_t getDataLen();
	virtual void reset() = 0;
	virtual void process_buffer(float* real, float* imaginary, size_t readcount) = 0;
};

class MultichannelSignalProcessor
{
private:
	size_t datalen;
	size_t channels;
	SignalProcessor** processors;
public:
	MultichannelSignalProcessor(size_t datalen, size_t channels, SignalProcessor** processors);
	size_t getDataLen();
	void reset();
	void process_buffer(float* real, float* imaginary, size_t readcount);
};

class NoProcessor : public SignalProcessor
{
public:
	NoProcessor(size_t datalen);
	void reset();
	void process_buffer(float* real, float* imaginary, size_t readcount);
};
