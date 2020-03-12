#pragma once

#define _CRT_SECURE_NO_WARNINGS
#include "SignalProcessor.h"
#include <iostream>

class CsvFileWriter : public SignalProcessor
{
private:
	std::string filename;
	FILE* file;
	size_t count;
public:
	CsvFileWriter(AbstractSignalProcessor* p, BitMask channels_to_process, std::string filename);
	~CsvFileWriter();
	int init(size_t max_buffer_size, size_t channels);
	void process_buffer(SignalBuffer_t* buffer);
};