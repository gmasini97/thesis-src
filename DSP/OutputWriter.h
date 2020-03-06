#pragma once

class OutputWriter
{
public:
	virtual void write_buffer(float* real, float* imaginary, size_t datalen) = 0;
	virtual void close() = 0;
};