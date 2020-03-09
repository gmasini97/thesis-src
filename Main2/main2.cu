#include "DSP.h"
#include "DFT.h"
#include "FFT.h"
#include "IFTT.h"
#include "ToPolar.h"
#include "CSVOutputWriter.h"
#include "WavOutputWriter.h"
#include "CUDADFT.cuh"
#include "SignalProcessorChain.h"
#include "Convolver.h"
#include <sndfile.h>
#include <iostream>

#define filter_size 16

using namespace DSP;

int main(int argc, char** argv)
{

	SNDFILE* infile, *convfile;

	SF_INFO sf_info, sf_info2;

	memset(&sf_info, 0, sizeof(sf_info));

	infile = sf_open_fd(0, SFM_READ, &sf_info, true);
	convfile = sf_open("D:\\avg16.wav", SFM_READ, &sf_info2);

	SignalBuffer_t convsig = allocate_signal_buffer(filter_size);
	float* data = new float[filter_size];
	float* nodata = new float[filter_size]{0};
	size_t readcount = sf_read_float(convfile, data, filter_size);
	signal_buffer_from_floats(&convsig, data, nodata, 1, readcount);

	size_t channels = sf_info.channels;
	size_t datalen = atoi(argv[1]) * channels;
	size_t subProcDataLen = datalen / channels;

	cout << channels << endl;
	flush(cout);

	SignalProcessor** procs = new SignalProcessor * [channels];

	for (size_t i = 0; i < channels; i++) {
		SignalProcessor** arr = new SignalProcessor * [2];
		arr[0] = new Convolver(subProcDataLen, convsig);
		//arr[1] = new CSVOutputWriter(subProcDataLen, "D:\\csv.csv");
		arr[1] = new WavOutputWriter(subProcDataLen, sf_info, "D:\\out.wav");
		SignalProcessor* chain = new SignalProcessorChain(subProcDataLen, arr, 2);
		procs[i] = chain;
	}

	std::cout << runOnFile(infile, &sf_info, datalen, procs) << std::endl;

	for (size_t i = 0; i < channels; i++) {
		delete procs[i];
	}
	delete[] procs;

	return 0;
}