#include "pch.h"
#include "LowPassFilter.h"

namespace DSP {

	LPFilterAvg::LPFilterAvg(size_t datalen, size_t points) : SignalProcessor(datalen)
	{
		this->points = points;
	}

	void LPFilterAvg::process_buffer(float* data, int readcount, int channels)
	{

	}

	void LPFilterAvg::reset()
	{

	}

}