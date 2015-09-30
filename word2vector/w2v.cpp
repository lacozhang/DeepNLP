#include <iostream>
#include <fstream>
#include <map>
#include <random>
#include <Eigen/Dense>
#include "../datalib/text.h"

void print_usage(){
	std::cout << "w2v.exe options" << std::endl;
	std::cout << "  -s : source plain text file" << std::endl
		<< "  -w : size of the generated vector" << std::endl
		<< "  -o : name of generated model" << std::endl
		<< "  -e : number of epochs" << std::endl
		<< "  -c : size of context window" << std::endl
		<< "  -n : negative sampling size" << std::endl;
}

void BuildWordDist(Eigen::RowVectorXf& dist, std::map<std::string, size_t>& word2id, std::vector<Sentence>& corpus){

	dist.resize(word2id.size());
	dist.setZero();
	
	for (Sentence& sent : corpus){
		for (size_t idx = 0; idx < sent.length(); ++idx){
			size_t wordidx = word2id[sent.word(idx)];
			dist.coeffRef(wordidx) += 1;
		}
	}

	dist /= dist.sum();

	dist.array().pow(0.75);

	dist /= dist.sum();

	for (size_t i = 1; i < dist.cols(); ++i){
		dist.coeffRef(i) += dist.coeff(i - 1);
	}
}

void saveModel(std::string modelname, std::map<std::string, size_t>& word2id, Eigen::MatrixXf& inner, Eigen::MatrixXf& outer){

	std::ofstream sink(modelname.c_str());

	if (!sink.good()){
		std::cerr << "Error opening file " << modelname << std::endl;
		std::exit(-1);
	}

	sink << word2id.size() << std::endl;
	for (std::pair<const std::string, size_t>& kv : word2id){
		sink << kv.first << "\t" << kv.second << std::endl;
	}

	sink << "inner vector" << std::endl;
	for (size_t i = 0; i < inner.rows(); ++i){
		sink << inner.row(i) << std::endl;
	}

	sink << "outer vector" << std::endl;
	for (size_t i = 0; i < outer.rows(); ++i){
		sink << outer.row(i) << std::endl;
	}
}

size_t sampling(Eigen::RowVectorXf& dist, double val){

	size_t low = 0, high = dist.cols() - 1;
	size_t mid = (low + high) / 2;

	while (low < high){
		
		if (dist[mid] > val){
			if (mid == 0){
				return 0;
			}
			else if (dist[mid - 1] < val){
				return mid;
			}
			else {
				high = mid;
			}
		}
		else {
			if (mid == dist.cols() - 1){
				return dist.cols() - 1;
			}
			else if (dist[mid + 1] > val){
				return mid + 1;
			}
			else {
				low = mid;
			}
		}

		mid = (low + high) / 2;
	}
}

void BuildLexicon(std::map<std::string, size_t>& word2id,
	std::map<size_t, std::string>& id2word,
	std::vector<Sentence>& corpus){

	word2id.clear();
	id2word.clear();

	for (Sentence& sent : corpus){
		for (size_t i = 0; i < sent.length(); ++i){
			if (word2id.count(sent.word(i)) > 0){
				continue;
			}
			else {
				int idx = word2id.size();
				word2id[sent.word(i)] = idx;
				id2word[idx] = sent.word(i);
			}
		}
	}
}

int main(int argc, char* argv[]){

	std::string textfile;
	int vecsize = 0;
	int epochs = 0;
	int context = 0;
	int negsamples = 0;
	std::string sinkfile;

	if (argc < 2){
		print_usage();
	}

	for (int i = 1; i < argc; ++i){
		switch (argv[i][1])
		{
		case 's':
			textfile = argv[++i];
			break;
		case 'w':
			vecsize = std::atoi(argv[++i]);
			break;
		case 'o':
			sinkfile = argv[++i];
			break;
		case 'e':
			epochs = std::atoi(argv[++i]);
			break;
		case 'c':
			context = std::atoi(argv[++i]);
			break;
		case 'n':
			negsamples = std::atoi(argv[++i]);
			break;
		default:
			print_usage();
			std::exit(-1);
		}
	}

	std::cout << "source text file : " << textfile << std::endl
		<< "vector size      : " << vecsize << std::endl
		<< "model file       : " << sinkfile << std::endl
		<< "#epochs          : " << epochs << std::endl
		<< "context size     : " << context << std::endl
		<< "negative samples : " << negsamples << std::endl;

	std::vector<Sentence> corpus;
	std::map<std::string, size_t> word2id;
	std::map<size_t, std::string> id2word;
	Eigen::RowVectorXf unidist;

	LoadData(textfile, corpus);
	BuildLexicon(word2id, id2word, corpus);
	BuildWordDist(unidist, word2id, corpus);

	int wordcounts = word2id.size();
	std::cout << "#words : " << wordcounts << std::endl;

	srand(0);
	Eigen::MatrixXf innerVector, outerVector;
	innerVector.resize(wordcounts, vecsize);
	outerVector.resize(wordcounts, vecsize);

	innerVector.setRandom();
	outerVector.setRandom();

	std::cout << "inner " << std::endl;
	std::cout << innerVector << std::endl;
	std::cout << "outer " << std::endl;
	std::cout << outerVector << std::endl;

	std::uniform_real_distribution<double> wordselector(0.0, 1.0);
	std::default_random_engine engine(0);

	// training with word embeddings.
	for (int i = 0; i < epochs; ++i){
		for (Sentence& sent : corpus){
			for (int idx = 0; idx < sent.length(); ++idx){
				size_t leftIdx = Sentence::GetPos(sent.length(), idx, -context);
				size_t rightIdx = Sentence::GetPos(sent.length(), idx, context);

				std::vector<size_t> posindx;
				std::vector<size_t> negindx;

				for (size_t contextidx = leftIdx; contextidx <= rightIdx; ++contextidx){
					if (contextidx == idx)
						continue;
					posindx.push_back(word2id[sent.word(contextidx)]);
				}

				for (int negcnt = 0; negcnt < negsamples; ++ negcnt){

					size_t selwordidx = 0;
					do{
						double randomvalue = wordselector(engine);
						selwordidx = sampling(unidist, randomvalue);
						bool realneg = true;
						for (size_t posidx : posindx){
							if (posidx == selwordidx)
								realneg = false;
						}

						for (size_t negidx : negindx){
							if (negidx == selwordidx)
								realneg = false;
						}

						if (realneg)
							break;
					} while (true);

					negindx.push_back(selwordidx);
				}

				Eigen::MatrixXf posoutervec(posindx.size(), vecsize);
				Eigen::MatrixXf posgrad(posindx.size(), vecsize);

				Eigen::MatrixXf negoutervec(negindx.size(), vecsize);
				Eigen::MatrixXf neggrad(negindx.size(), vecsize);

				// update the information
				for (int posIdx = 0; posIdx < posindx.size(); ++posIdx){
					posoutervec.row(posIdx) = outerVector.row(posindx[posIdx]);
				}

				for (int negIdx = 0; negIdx < negindx.size(); ++negIdx){
					negoutervec.row(negIdx) = outerVector.row(negindx[negIdx]);
				}

				// initialize the vector
				Eigen::RowVectorXf centerInnervec = innerVector.row(word2id[sent.word(idx)]);

				Eigen::RowVectorXf posrawscore = posoutervec * centerInnervec.transpose();
				Eigen::RowVectorXf negrawscore = negoutervec * centerInnervec.transpose();

				double expsum = posrawscore.array().exp().sum() + negrawscore.array().exp().sum();

				Eigen::RowVectorXf posprob = posrawscore.array().exp() / expsum;
				Eigen::RowVectorXf negprob = negrawscore.array().exp() / expsum;

				// calculate the gradient of center word inner vector.
				Eigen::RowVectorXf centerInnergrad = posoutervec.colwise().sum() - posprob * posoutervec;
				centerInnergrad *= -1;

				// calculate the gradient of surrounding word outer vector
				for (size_t posindx = 0; posindx < posgrad.rows(); ++posindx){
					posgrad.row(posindx) = -1 * (centerInnervec - posprob[posindx] * posgrad.rows() * centerInnervec);					
				}

				// calculate the negative samples
				for (size_t negidx = 0; negidx < neggrad.rows(); ++negidx){
					neggrad.row(negidx) = negprob[negidx] * centerInnervec;
				}

				// update the gradient
				size_t centerWordId = word2id[sent.word(idx)];
				innerVector.row(centerWordId) -= 0.1 * centerInnergrad;

				for (size_t posIdx = 0; posIdx < posindx.size(); ++posIdx){
					size_t wordId = posindx[posIdx];
					outerVector.row(wordId) -= posgrad.row(posIdx);
				}

				for (size_t negIdx = 0; negIdx < negindx.size(); ++negIdx){
					size_t wordId = negindx[negIdx];
					outerVector.row(wordId) -= neggrad.row(negIdx);
				}

				std::cout << "Update Vector : " << std::endl;
				std::cout << "inner " << std::endl;
				std::cout << innerVector << std::endl;
				std::cout << "outer " << std::endl;
				std::cout << outerVector << std::endl;

			}
		}
	}

	saveModel(sinkfile, word2id, innerVector, outerVector);
	return 0;
}