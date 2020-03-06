#include "pch.h"
#include "framework.h"
#include "DSP.h"
#include "DSP.h"

namespace DSP {
	int runOnFile(SNDFILE* infile, SF_INFO* sf_info, size_t bufferLen, MultichannelSignalProcessor* msp, OutputWriter* outputWriter) {

		float* data = (float*)malloc(bufferLen * sizeof(float));
		float* nodata = (float*)malloc(bufferLen * sizeof(float));

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
			outputWriter->write_buffer(data, nodata, readcount);
		}

		outputWriter->close();
		sf_close(infile);

		free(data);

		return 0;
	}
}
