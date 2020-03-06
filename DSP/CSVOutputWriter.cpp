#include "pch.h"
#include "CSVOutputWriter.h"


CSVOutputWriter::CSVOutputWriter(size_t datalen, const char* filename) : OutputWriter(datalen)
{
	this->filename = filename;
	this->file = fopen(filename, "w");
	this->count = 0;
}
CSVOutputWriter::~CSVOutputWriter()
{
	fclose(this->file);
}

void CSVOutputWriter::reset()
{
	fseek(this->file, 0, SEEK_SET);
	this->count = 0;
}

void CSVOutputWriter::process_buffer(float* real, float* imaginary, size_t datalen)
{
	for (size_t i = 0; i < datalen; i++)
	{
		fprintf(this->file, "%lli,%f,%f\n", count++, real[i], imaginary[i]);
	}
}