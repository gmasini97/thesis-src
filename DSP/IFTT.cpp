#include "pch.h"
#include "IFTT.h"

void ifft(float* real, float* imaginary, size_t size)
{
	for (size_t k = 0; k < size; k++)
	{
		imaginary[k] = -imaginary[k];
	}

	fft(real, imaginary, size);

	for (size_t i = 0; i < size; i++)
	{
		real[i] = real[i] / size;
		imaginary[i] = -imaginary[i] / size;
	}
}

IFFTProcessor::IFFTProcessor(size_t datalen) : SignalProcessor(datalen)
{
}
IFFTProcessor::~IFFTProcessor()
{
}

void IFFTProcessor::reset()
{
}

void IFFTProcessor::process_buffer(float* real, float* imaginary, size_t readcount)
{
	ifft(real, imaginary, readcount);
}