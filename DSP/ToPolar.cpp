#include "pch.h"
#include "ToPolar.h"

void polar(float* re, float* im)
{
	float magnitude = sqrt(*re * *re + *im * *im);
	float phase = atan2(*im, *re);

	*re = magnitude;
	*im = phase;
}

ToPolar::ToPolar(size_t datalen) : SignalProcessor(datalen)
{
}
ToPolar::~ToPolar()
{
}

void ToPolar::reset()
{
}

void ToPolar::process_buffer(float* real, float* imaginary, size_t readcount)
{
	for (size_t i = 0; i < readcount; i++)
	{
		polar(real + i, imaginary + i);
	}
}