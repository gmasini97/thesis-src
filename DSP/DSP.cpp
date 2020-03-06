#include "pch.h"
#include "framework.h"
#include "DSP.h"
#include "DSP.h"

namespace DSP {
	int runOnFile(SNDFILE* infile, SF_INFO* sf_info, size_t bufferLen, MultichannelSignalProcessor* msp) {

		float* data = new float[bufferLen];
		float* nodata = new float[bufferLen];

		for (size_t i = 0 ; i < bufferLen ; i++)
			nodata[i] = 0;

		msp->reset();

		int readcount;

		if (!(infile) || !(msp))
		{
			cout << "Invalid inputs" << endl;
			return 1;
		}

		while ((readcount = sf_read_float(infile, data, bufferLen)))
		{
			msp->process_buffer(data, nodata, readcount);
		}

		sf_close(infile);

		delete[] data;

		return 0;
	}
}
