#include "DSP.h"
#include "DFT.h"
#include "FFT.h"
#include "IFTT.h"
#include "ToPolar.h"
#include "CSVOutputWriter.h"
#include <iostream>
#include "CUDADFT.h"

using namespace DSP;

int main(int argc, char** argv)
{
	SNDFILE* infile, * outfile = NULL;

	SF_INFO sf_info;

	memset(&sf_info, 0, sizeof(sf_info));


	infile = sf_open_fd(0, SFM_READ, &sf_info, true);

	size_t channels = sf_info.channels;
	size_t datalen = atoi(argv[1]) * channels;
	size_t subProcDataLen = datalen / channels;
	SignalProcessor** procs = new SignalProcessor * [channels];
	for (size_t i = 0; i < channels; i++) {
		SignalProcessor** arr = new SignalProcessor * [3];
		arr[0] = new CUDADFT(subProcDataLen);
		//arr[0] = new DFTProcessor(subProcDataLen);
		arr[1] = new NoProcessor(subProcDataLen);
		arr[2] = new CSVOutputWriter(subProcDataLen, "D:\\csv.csv");
		arr[3] = new ToPolar(subProcDataLen);
		arr[4] = new CSVOutputWriter(subProcDataLen, "D:\\polar.csv");
		SignalProcessor* chain = new SignalProcessorChain(subProcDataLen, arr, 3);
		procs[i] = chain;
	}
	MultichannelSignalProcessor* msp = new MultichannelSignalProcessor(datalen, channels, procs);

	std::cout << runOnFile(infile, &sf_info, datalen, msp) << std::endl;

	delete msp;

	return 0;
}