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
	CSVOutputWriter(const char* filename);
	void write_buffer(float* real, float* imaginary, size_t datalen);
	void close();
};