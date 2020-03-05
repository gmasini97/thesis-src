#pragma once
#define _USE_MATH_DEFINES
#include "SignalProcessor.h"
#include <cmath>

void butterfly_calculation(float* ar, float* ai, float* br, float* bi, float ur, float ui);
void bitreverse_sort(float* re, float* im, size_t datalen);
void process(float* re, float* im, size_t datalen);

class FFTProcessor : public SignalProcessor
{
private:
	float* im;
	float* re;
public:
	FFTProcessor(size_t datalen);
	void reset();
	void process_buffer(float* data, size_t readcount);
};
