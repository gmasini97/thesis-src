#include "DSP.h"

namespace DSP {

	int any_processor_has_extra(SignalProcessor** sp, size_t channels)
	{
		for (size_t i = 0; i < channels; i++)
		{
			if (sp[i]->has_extra_samples())
				return 1;
		}
		return 0;
	}

	long long int runOnFile(SNDFILE* infile, SF_INFO* sf_info, size_t bufferLen, SignalProcessor** sp) {

		float* data = new float[bufferLen];
		float* nodata = new float[bufferLen];
		size_t readcount;

		SignalBuffer_t signalBuffer = allocate_signal_buffer(bufferLen);
		size_t channels = sf_info->channels;

		for (size_t i = 0 ; i < bufferLen ; i++)
			nodata[i] = 0;

		for (size_t i = 0; i < channels; i++)
		{
			sp[i]->reset();
		}

		if (!(infile) || !(sp))
		{
			cout << "Invalid inputs" << endl;
			return 1;
		}

		auto start = std::chrono::high_resolution_clock::now();

		while ((readcount = sf_read_float(infile, data, bufferLen)))
		{
			signal_buffer_from_floats(&signalBuffer, data, nodata, channels, readcount);
			for (size_t i = 0; i < channels; i++)
			{
				sp[i]->process_buffer(&signalBuffer, i);
			}
		}

		while (any_processor_has_extra(sp, channels))
		{
			signal_buffer_from_floats(&signalBuffer, nodata, nodata, channels, bufferLen);
			signalBuffer.size = 0;
			for (size_t i = 0; i < channels; i++)
			{
				SignalProcessor* s = sp[i];
				if (s->has_extra_samples())
					s->get_extra_samples(&signalBuffer, i);
			}
		}

		auto end = std::chrono::high_resolution_clock::now();

		sf_close(infile);

		delete[] data;
		delete[] nodata;

		return std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
	}
}
