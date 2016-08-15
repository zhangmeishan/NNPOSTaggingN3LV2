/*
 * TNNClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_TNNClassifier_H_
#define SRC_TNNClassifier_H_

#include <iostream>

#include "Example.h"
#include "Metric.h"
#include "N3L.h"

using namespace nr;
using namespace std;

struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 256;
	const static int max_character_length = 20;

public:
	// node instances
	vector<LookupNode> word_inputs;
	vector<vector<LookupNode> > char_inputs;

	vector<WindowBuilder> char_window;
	vector<vector<UniNode> > char_hidden1;
	vector<MaxPoolNode> char_hidden2;

	vector<ConcatNode> word_char_concat;
	WindowBuilder word_window;
	vector<UniNode> word_hidden1;
	vector<LinearNode> output;

	// node pointers
public:
	ComputionGraph() : Graph(){
		createNodes(max_sentence_length, max_character_length);
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length, int char_length){
		char_inputs.resize(sent_length);
		char_window.resize(sent_length);
		char_hidden1.resize(sent_length);
		char_hidden2.resize(sent_length);
		for (int idx = 0; idx < sent_length; idx++){
			char_inputs[idx].resize(char_length);
			char_window[idx].resize(char_length);
			char_hidden1[idx].resize(char_length);
		}

		word_inputs.resize(sent_length);
		word_window.resize(sent_length);
		word_char_concat.resize(sent_length);
		word_hidden1.resize(sent_length);
		output.resize(sent_length);
	}

	inline void clear(){
		Graph::clear();
		for (int idx = 0; idx < char_inputs.size(); idx++){
			char_inputs[idx].clear();
			char_window[idx].clear();
			char_hidden1[idx].clear();
		}
		char_inputs.clear();
		char_window.clear();
		char_hidden1.clear();
		char_hidden2.clear();

		word_inputs.clear();
		word_window.clear();
		word_char_concat.clear();
		word_hidden1.clear();
		output.clear();
	}

public:
	inline void initial(LookupTable& words, LookupTable& chars, UniParams& tanhchar_project,
		UniParams& tanh_project, UniParams& olayer_linear, int wordcontext, int charcontext){
		for (int idx = 0; idx < max_sentence_length; idx++) {
			word_inputs[idx].setParam(&words);
			word_hidden1[idx].setParam(&tanh_project);
			word_hidden1[idx].setFunctions(&tanh, &tanh_deri);
			output[idx].setParam(&olayer_linear);
			for (int idy = 0; idy < max_character_length; idy++) {
				char_inputs[idx][idy].setParam(&chars);
				char_hidden1[idx][idy].setParam(&tanhchar_project);
				char_hidden1[idx][idy].setFunctions(&tanh, &tanh_deri);
			}			
			char_window[idx].setContext(charcontext);
		}	
		word_window.setContext(wordcontext);
	}



public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const vector<Feature>& features, bool bTrain = false){
		clearValue(); // compute is a must step for train, predict and cost computation
		int seq_size = features.size();
		//forward
		// neural networks based on characters
		for (int idx = 0; idx < seq_size; idx++) {
			const Feature& feature = features[idx];
			int char_num = feature.chars.size();
			if (char_num > max_character_length){
				char_num = max_character_length;
			}

			//char input		
			for (int idy = 0; idy < char_num; idy++) {
				char_inputs[idx][idy].forward(feature.chars[idy]);
				execs.push_back(&char_inputs[idx][idy]);
			}

			//windowlized
			char_window[idx].forward(getPNodes(char_inputs[idx], char_num));
			char_window[idx].traverseNodes(execs);

			// feed-forward
			for (int idy = 0; idy < char_num; idy++) {
				char_hidden1[idx][idy].forward(&(char_window[idx]._outputs[idy]));
				execs.push_back(&char_hidden1[idx][idy]);
			}

			//pooling 
			char_hidden2[idx].forward(getPNodes(char_hidden1[idx], char_num));
			execs.push_back(&char_hidden2[idx]);
		}

		// word-level neural networks
		for (int idx = 0; idx < seq_size; idx++) {
			const Feature& feature = features[idx];
			//input
			word_inputs[idx].forward(feature.words[0]);
			execs.push_back(&word_inputs[idx]);

			//concat
			word_char_concat[idx].forward(&word_inputs[idx], &char_hidden2[idx]);
			execs.push_back(&word_char_concat[idx]);
		}

		//windowlized
		word_window.forward(getPNodes(word_char_concat, seq_size));
		word_window.traverseNodes(execs);

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			word_hidden1[idx].forward(&(word_window._outputs[idx]));
			execs.push_back(&word_hidden1[idx]);

			//output
			output[idx].forward(&(word_hidden1[idx]));
			execs.push_back(&output[idx]);
		}
	}

};

//A native neural network classfier using only word embeddings

class TNNClassifier {
public:
	TNNClassifier() {
		_dropOut = 0.0;
		_pcg = NULL;
	}

	~TNNClassifier() {
		if (_pcg != NULL)
			delete _pcg;
		_pcg = NULL;
	}

public:
	LookupTable _words;
	LookupTable _chars;

	int _wordcontext, _wordwindow;
	int _wordSize;
	int _wordDim;

	int _charcontext, _charwindow;
	int _charSize;
	int _charDim;
	int _char_outputSize;
	int _char_inputSize;

	int _hiddensize;
	int _inputsize, _token_representation_size;

	UniParams _olayer_linear;
	UniParams _tanhchar_project;
	UniParams _tanh_project;

	SoftMaxLoss _loss;

	int _labelSize;

	Metric _eval;

	dtype _dropOut;

	ModelUpdate _ada;

	ComputionGraph *_pcg;

	CheckGrad _checkgrad;

public:
	//embeddings are initialized before this separately.
	inline void init(int wordcontext, int charcontext, int charhiddensize, int hiddensize, int labelSize) {
		if (_words.nVSize == 0 || _chars.nVSize == 0){
			std::cout << "Please initialize embeddings before this" << std::endl;
			return;
		}
		_wordcontext = wordcontext;
		_wordwindow = 2 * _wordcontext + 1;
		_wordDim = _words.nDim;

		_charcontext = charcontext;
		_charwindow = 2 * _charcontext + 1;
		_charDim = _chars.nDim;

		_char_inputSize = _charwindow * _charDim;
		_char_outputSize = charhiddensize;

		_labelSize = labelSize;
		_hiddensize = hiddensize;
		_token_representation_size = _wordDim + _char_outputSize;
		_inputsize = _wordwindow * _token_representation_size;


		_tanh_project.initial(_hiddensize, _inputsize, true, 100);
		_tanhchar_project.initial(_char_outputSize, _char_inputSize, true, 200);
		_olayer_linear.initial(_labelSize, _hiddensize, false, 300);

		//ada
		_words.exportAdaParams(_ada);
		_chars.exportAdaParams(_ada);
		_tanh_project.exportAdaParams(_ada);
		_tanhchar_project.exportAdaParams(_ada);
		_olayer_linear.exportAdaParams(_ada);

		_pcg = new ComputionGraph();
		_pcg->initial(_words, _chars, _tanhchar_project, _tanh_project, _olayer_linear, _wordcontext, _charcontext);

		//check grad
		_checkgrad.add(&(_words.E), "_words.E");
		_checkgrad.add(&(_chars.E), "_chars.E");
		_checkgrad.add(&(_tanh_project.W), "_tanh_project.W");
		_checkgrad.add(&(_tanh_project.b), "_tanh_project.b");
		_checkgrad.add(&(_tanhchar_project.W), "_tanhchar_project.W");
		_checkgrad.add(&(_tanhchar_project.b), "_tanhchar_project.b");
		_checkgrad.add(&(_olayer_linear.W), "_olayer_linear.W");
		//_checkgrad.add(&(_olayer_linear.b), "_olayer_linear.b");
	}


	inline dtype train(const vector<Example>& examples, int iter) {
		_eval.reset();

		int example_num = examples.size();
		dtype cost = 0.0;

		static vector<PMat> tpmats;

		for (int count = 0; count < example_num; count++) {
			const Example& example = examples[count];

			//forward
			_pcg->forward(example.m_features, true); 

			//loss function
			int seq_size = example.m_features.size();
			for (int idx = 0; idx < seq_size; idx++) {
				cost += _loss.loss(&(_pcg->output[idx]), example.m_labels[idx], _eval, example_num);
			}

			// backward, which exists only for training 
			_pcg->backward();
		}

		if (_eval.getAccuracy() < 0) {
			std::cout << "strange" << std::endl;
		}

		return cost;
	}

	inline void predict(const vector<Feature>& features, vector<int>& results) {
		_pcg->forward(features);
		int seq_size = features.size();
		results.resize(seq_size);
		for (int idx = 0; idx < seq_size; idx++) {
			_loss.predict( &(_pcg->output[idx]), results[idx]);
		}
	}

	inline dtype cost(const Example& example){
		_pcg->forward(example.m_features); //forward here

		int seq_size = example.m_features.size();

		dtype cost = 0.0;
		//loss function
		for (int idx = 0; idx < seq_size; idx++) {
			cost += _loss.cost(&(_pcg->output[idx]), example.m_labels[idx], 1);
		}

		return cost;
	}


	void updateModel() {
		_ada.update();
	}

	void checkgrad(const vector<Example>& examples, int iter){
		ostringstream out;
		out << "Iteration: " << iter;
		_checkgrad.check(this, examples, out.str());
	}

	void writeModel();

	void loadModel();



public:
	inline void resetEval() {
		_eval.reset();
	}

	inline void setDropValue(dtype dropOut) {
		_dropOut = dropOut;
	}

	inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps){
		_ada._alpha = adaAlpha;
		_ada._eps = adaEps;
		_ada._reg = nnRegular;
	}

};

#endif /* SRC_TNNClassifier_H_ */
