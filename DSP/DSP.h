#pragma once

#include <iostream>
#include <string>
#include <sndfile.h>
#include "SignalProcessor.h"
#include "OutputWriter.h"

using namespace std;

namespace DSP {
	int runOnFile(SNDFILE* infile, SF_INFO* sf_info, size_t bufferLen, MultichannelSignalProcessor* msp, OutputWriter* outputWriter);

	//int runWithStdInOut(size_t bufferLen, SignalProcessor* processor);
}

