#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include "SignalProcessor.h"
#include "CsvFileWriter.h"
#include "Convolver.h"
#include "SndFileWriter.h"
#include "DFT.h"
#include <sstream>
#include <fstream>

#define MAX_LINE_SIZE 256

AbstractSignalProcessor* build_fx_chain(std::string filename);
AbstractSignalProcessor* build_fx_chain_rec(std::ifstream* file);
AbstractSignalProcessor* create_from_line(std::string line, AbstractSignalProcessor* next);