#pragma once

#include <iostream>
#include <string>
#include <sndfile.h>
#include "SignalProcessor.h"
#include "OutputWriter.h"
#include <chrono>

using namespace std;

namespace DSP {
	long long int runOnFile(SNDFILE* infile, SF_INFO* sf_info, size_t bufferLen, MultichannelSignalProcessor* msp);

	//int runWithStdInOut(size_t bufferLen, SignalProcessor* processor);
}

