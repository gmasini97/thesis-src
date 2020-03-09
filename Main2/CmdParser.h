#pragma once

#include <sndfile.h>
#include "SignalProcessor.h"

struct ProgramContext_t
{
	SNDFILE* infile;
	size_t buffer_length;
	SignalProcessor** processors;
};

int parse_parameters(ProgramContext_t* context, int argc, char** argv)
{
	if (argc < 2)
		return 1;

	SignalProcessor** processors = new SignalProcessor*[(size_t)argc-2];

	for (int i = 0; i < argc - 1; i++)
	{
		char* arg = argv[i];

	}
}
