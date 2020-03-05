#include "pch.h"
#include "framework.h"
#include "DSP.h"
#include "SignalProcessor.h"

namespace DSP {
	int runWithFiles(SNDFILE* infile, SNDFILE* outfile, SF_INFO* sf_info, size_t bufferLen, MultichannelSignalProcessor* msp) {

		float* data = (float*)malloc(bufferLen * sizeof(float));

		msp->reset();

		int readcount;

		if (!(infile) || !(outfile) || !(msp))
		{
			cout << "Invalid inputs" << endl;
			return 1;
		}

		while ((readcount = sf_read_float(infile, data, bufferLen)))
		{
			msp->process_buffer(data, readcount);
			sf_write_float(outfile, data, readcount);
		}

		sf_close(outfile);
		sf_close(infile);

		free(data);

		return 0;
	}
}
