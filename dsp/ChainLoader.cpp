#include "ChainLoader.h"

using namespace std;

int chain_loader_read_str(istringstream &ss, string &s)
{
	if (!getline(ss, s, ',')) {
		cerr << "err reading string, " << s << endl;
		return 0;
	}
	return 1;
}

int chain_loader_read_float(istringstream &ss, float &f)
{
	string s;
	if (!chain_loader_read_str(ss, s)) {
		return 0;
	}
	f = stof(s);
	return 1;
}

int chain_loader_read_int(istringstream& ss, int &f)
{
	string s;
	if (!chain_loader_read_str(ss, s)) {
		return 0;
	}
	f = stoi(s);
	return 1;
}

int chain_loader_read_hex_int(istringstream &ss, BitMask &f)
{
	string s;
	if (!chain_loader_read_str(ss, s)) {
		return 0;
	}
	istringstream convert(s);
	BitMask h;
	convert >> std::hex >> h;
	f = h;
	return 1;
}

AbstractSignalProcessor* build_fx_chain(std::string filename, AbstractSignalProcessor* previous)
{
	string line;
	ifstream file(filename);

	AbstractSignalProcessor* a = build_fx_chain_rec(&file, previous);

	file.close();

	return a;
}

AbstractSignalProcessor* build_fx_chain_rec(ifstream* file, AbstractSignalProcessor* previous)
{
	string line;
	if (!getline(*file, line)) {
		return previous;
	}
	AbstractSignalProcessor* asp = create_from_line(line, previous);
	return build_fx_chain_rec(file, asp);
}

AbstractSignalProcessor* create_from_line(string line, AbstractSignalProcessor* previous)
{
	if (line.size() <= 0)
		return NULL;
	istringstream ss(line);
	string name;
	BitMask mask;

	if (!chain_loader_read_str(ss, name))
		return NULL;

	if (!chain_loader_read_hex_int(ss, mask))
		return NULL;

	if (name == "csvout")
	{
		string filename;
		if (!chain_loader_read_str(ss, filename))
			return NULL;
		return new CsvFileWriter(previous, mask, filename);
	} else
	if (name == "wavout")
	{
		int channels;
		if (!chain_loader_read_int(ss, channels))
			return NULL;

		string filename;
		if (!chain_loader_read_str(ss, filename))
			return NULL;

		SF_INFO info;
		memset(&info, 0, sizeof(info));
		info.channels = channels;
		info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT | SF_ENDIAN_FILE;
		info.samplerate = 44100;

		return new SndFileWriter(previous, mask, filename, info);
	}
	else
	if (name == "conv")
	{
		int size;
		if (!chain_loader_read_int(ss, size))
			return NULL;

		string filename;
		if (!chain_loader_read_str(ss, filename))
			return NULL;

		return create_convolver_from_file(previous, mask, filename, size);
	}
	else
	if (name == "fftconv")
	{
		int size;
		if (!chain_loader_read_int(ss, size))
			return NULL;

		string filename;
		if (!chain_loader_read_str(ss, filename))
			return NULL;

		return create_fftconvolver_from_file(previous, mask, filename, size);
	}
	else
	if (name == "gain")
	{
		float reg, img;
		if (!chain_loader_read_float(ss, reg))
			return NULL;
		if (!chain_loader_read_float(ss, img))
			return NULL;
		return new GainProcessor(previous, mask, reg, img);
	}
	else
	if (name == "dft")
	{
		int size;
		if (!chain_loader_read_int(ss, size))
			return NULL;
		return new DFTProcessor(previous,mask,size);
	}
	else
	if (name == "fft")
	{
		int size;
		if (!chain_loader_read_int(ss, size))
			return NULL;
		return new FFTProcessor(previous, mask,size);
	}
	else
	if (name == "ifft")
	{
		return new IFFTProcessor(previous, mask);
	}
	else
	if (name == "cudadft")
	{
		return new CUDADFT(previous, mask);
	}
	else
	if (name == "cudaconv")
	{
		int size;
		if (!chain_loader_read_int(ss, size))
			return NULL;

		string filename;
		if (!chain_loader_read_str(ss, filename))
			return NULL;

		return create_cuda_convolver_from_file(previous, mask, filename, size);
	}

	return NULL;
}