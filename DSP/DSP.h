#pragma once

#include <iostream>
#include <string>
#include <sndfile.h>
#include "SignalProcessor.h"

using namespace std;

namespace DSP {
	int runWithFiles(SNDFILE* infile, SNDFILE* outfile, SF_INFO* sf_info, size_t bufferLen, SignalProcessor* processor);

	int runWithStdInOut(size_t bufferLen, SignalProcessor* processor);
}

