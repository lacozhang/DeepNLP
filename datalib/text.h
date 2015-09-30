
#ifndef __TEXT_DATA_H__
#define __TEXT_DATA_H__

#include <string>
#include <vector>

class Sentence {
public:
	Sentence();
	Sentence(std::string);

	size_t length() const {
		return mWords_.size();
	}

	std::string& word(size_t idx){
		return mWords_.at(idx);
	}

	virtual ~Sentence();

	static size_t GetPos(size_t len, size_t idx, int offset);

private:
	std::vector<std::string> mWords_;
};

size_t GetPos(size_t len, size_t idx, int offset);

void LoadData(std::string filename, std::vector<Sentence>& corpus);

#endif // __TEXT_DATA_H__