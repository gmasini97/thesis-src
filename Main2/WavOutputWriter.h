#pragma once

#define _CRT_SECURE_NO_WARNINGS
#include "OutputWriter.h"
#include <sndfile.h>
#include <iostream>

class WavOutputWriter : public OutputWriter
{
private:
	const char* filename;
	SNDFILE* file;
	SF_INFO info;
	float* real, *imaginary;
public:
	WavOutputWriter(size_t datalen, SF_INFO templ, const char* filename);
	~WavOutputWriter();
	void reset();
	void process_buffer(SignalBuffer_t* buffer, size_t channel);
};