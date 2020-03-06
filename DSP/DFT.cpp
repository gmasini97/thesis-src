#include "pch.h"
#include "DFT.h"

void dft(float* real, float* imaginary, size_t size)
{
	float* re = new float[size];
	float* im = new float[size];

	float sr, si;

	for (size_t k = 0; k < size; k++)
	{
		re[k] = 0;
		im[k] = 0;
		for (size_t i = 0; i < size; i++)
		{
			sr = (float) cos(2 * M_PI * k * i / size);
			si = (float) -sin(2 * M_PI * k * i / size);

			re[k] += real[i] * sr - imaginary[i] * si;
			im[k] += real[i] * si + imaginary[i] * sr;
		}
	}

	for (size_t k = 0; k < size; k++)
	{
		real[k] = re[k];
		imaginary[k] = im[k];
	}

	delete[] re;
	delete[] im;
}

DFTProcessor::DFTProcessor(size_t datalen) : SignalProcessor(datalen)
{
}
DFTProcessor::~DFTProcessor()
{
}

void DFTProcessor::reset()
{
}

void DFTProcessor::process_buffer(float* real, float* imaginary, size_t readcount)
{
	dft(real, imaginary, readcount);
}