#pragma once
#include "SignalProcessor.h"
#define _USE_MATH_DEFINES
#include <cmath>

class ToPolar : public SignalProcessor
{
public:
	ToPolar(size_t datalen);
	~ToPolar();
	void process_buffer(SignalBuffer_t* buffer, size_t channel);
};