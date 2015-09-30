#include <fstream>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include "text.h"

Sentence::Sentence(){
	mWords_.clear();
}

Sentence::~Sentence(){

}

Sentence::Sentence(std::string line){
	mWords_.clear();
	boost::algorithm::split(mWords_, line, boost::algorithm::is_space());

	if (mWords_.size() == 0){
		std::cerr << "Error, empty line" << std::endl;
	}
}

size_t Sentence::GetPos(size_t len, size_t idx, int offset){

	int newIdx = idx + offset;
	if (newIdx < 0){
		return 0;
	}

	if (newIdx >= len){
		return len - 1;
	}

	return newIdx;
}


void LoadData(std::string filename, std::vector<Sentence>& corpus){

	std::string line;
	std::ifstream src(filename.c_str());

	if (!src.good()){
		std::cerr << "Error opening file " << filename << std::endl;
		std::exit(-1);
	}

	std::getline(src, line);

	while (src.good())
	{
		boost::algorithm::trim(line);

		if (line.length() == 0){
			std::getline(src, line);
			continue;
		}

		Sentence curr(line);
		corpus.push_back(std::move(curr));

		std::getline(src, line);
	}
}
