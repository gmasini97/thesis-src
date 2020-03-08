#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include "OutputWriter.h"
#include <iostream>

class CSVOutputWriter : public OutputWriter
{
private:
	const char* filename;
	FILE* file;
	size_t count;
public:
	CSVOutputWriter(size_t datalen, const char* filename);
	~CSVOutputWriter();
	void reset();
	void process_buffer(SignalBuffer_t* buffer, size_t channel);
};