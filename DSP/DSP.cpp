#include "pch.h"
#include "framework.h"
#include "DSP.h"
#include "SignalProcessor.h"

namespace DSP {
	int runWithFiles(SNDFILE* infile, SNDFILE* outfile, SF_INFO* sf_info, size_t bufferLen, SignalProcessor* processor) {

		float* data = (float*)malloc(bufferLen * sizeof(float));

		int readcount;

		if (!(infile) || !(outfile) || !(processor))
		{
			cout << "Invalid inputs" << endl;
			return 1;
		}

		while ((readcount = sf_read_float(infile, data, bufferLen)))
		{
			processor->process_buffer(data, bufferLen, readcount, sf_info->channels);
			sf_write_float(outfile, data, readcount);
		}

		sf_close(outfile);
		sf_close(infile);

		free(data);

		return 0;
	}



	int runWithStdInOut(size_t bufferLen, SignalProcessor* processor)
	{
		SNDFILE* infile, * outfile;

		SF_INFO sf_info;

		memset(&sf_info, 0, sizeof(sf_info));

		infile = sf_open_fd(0, SFM_READ, &sf_info, true);
		outfile = sf_open_fd(1, SFM_WRITE, &sf_info, true);

		return runWithFiles(infile, outfile, &sf_info, bufferLen, processor);
	}
}
