#include "pch.h"
#include "CSVOutputWriter.h"


CSVOutputWriter::CSVOutputWriter(const char* filename)
{
	this->filename = filename;
	this->file = fopen(filename, "w");
	this->count = 0;
}

void CSVOutputWriter::close()
{
	fclose(this->file);
}

void CSVOutputWriter::write_buffer(float* real, float* imaginary, size_t datalen)
{
	for (size_t i = 0; i < datalen; i++)
	{
		fprintf(this->file, "%i,%f,%f\n", count++, real[i], imaginary[i]);
	}
}