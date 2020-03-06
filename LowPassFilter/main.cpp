#include "DSP.h"
#include "DFT.h"
#include "FFT.h"
#include "CSVOutputWriter.h"

using namespace DSP;

int main()
{
	SNDFILE* infile, * outfile= NULL;

	SF_INFO sf_info;

	memset(&sf_info, 0, sizeof(sf_info));

	infile = sf_open_fd(0, SFM_READ, &sf_info, true);
	CSVOutputWriter* csv = new CSVOutputWriter("D:\\csv.csv");

	size_t channels = sf_info.channels;
	size_t datalen = 512;
	size_t subProcDataLen = datalen / channels;
	SignalProcessor** procs = (SignalProcessor**)malloc(sizeof(SignalProcessor*) * channels);
	for (size_t i = 0; i < channels; i++) {
		procs[i] = new FFTProcessor(subProcDataLen);
	}
	MultichannelSignalProcessor* msp = new MultichannelSignalProcessor(datalen, channels, procs);

	return runOnFile(infile, &sf_info, datalen, msp, csv);

}