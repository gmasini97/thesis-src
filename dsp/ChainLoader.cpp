#include "ChainLoader.h"

using namespace std;

AbstractSignalProcessor* build_fx_chain(std::string filename)
{
	string line;
	ifstream file(filename);

	AbstractSignalProcessor* a = build_fx_chain_rec(&file);

	file.close();

	return a;
}

AbstractSignalProcessor* build_fx_chain_rec(ifstream* file)
{
	string line;
	if (!getline(*file, line)) {
		return NULL;
	}
	AbstractSignalProcessor* next = build_fx_chain_rec(file);
	return create_from_line(line, next);
}

AbstractSignalProcessor* create_from_line(string line, AbstractSignalProcessor* next)
{
	if (line.size() <= 0)
		return NULL;
	istringstream ss(line);
	string name;
	string bitmask;

	if (!getline(ss, name, ',')) {
		cerr << "err creating processor" << endl;
		return NULL;
	}
	if (!getline(ss, bitmask, ',')) {
		cerr << "channels err creating processor " << name << endl;
		return NULL;
	}

	istringstream convert(bitmask);
	BitMask mask;
	convert >> std::hex >> mask;

	if (name == "csvout")
	{
		string filename;
		if (!getline(ss, filename, ',')) {
			cerr << "err creating processor, filename, " << name << endl;
		}
		return new CsvFileWriter(next, mask, filename);
	} else
	if (name == "wavout")
	{
		string channels_s;
		if (!getline(ss, channels_s, ',')) {
			cerr << "err creating processor, channels, " << name << endl;
		}
		size_t channels = stoi(channels_s);

		string filename;
		if (!getline(ss, filename, ',')) {
			cerr << "err creating processor, filename, " << name << endl;
		}
		SF_INFO info;
		memset(&info, 0, sizeof(info));
		info.channels = channels;
		info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT | SF_ENDIAN_FILE;
		info.samplerate = 44100;

		return new SndFileWriter(next, mask, filename, info);
	}
	else
	if (name == "conv")
	{
		string size_s;
		if (!getline(ss, size_s, ',')) {
			cerr << "err creating processor, size_s, " << name << endl;
		}
		size_t size = stoi(size_s);

		string filename;
		if (!getline(ss, filename, ',')) {
			cerr << "err creating processor, filename, " << name << endl;
		}
		return create_convolver_from_file(next, mask, filename, size);
	}
	else
	if (name == "dft")
	{
		return new DFTProcessor(next,mask);
	}

	return NULL;
}