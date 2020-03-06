#pragma once
#include "SignalProcessor.h"
#define _USE_MATH_DEFINES
#include <cmath>

class ToPolar : public SignalProcessor
{
public:
	ToPolar(size_t datalen);
	~ToPolar();
	void reset();
	void process_buffer(float* real, float* imaginary, size_t readcount);
};