/*
 * SparseClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_SparseClassifier_H_
#define SRC_SparseClassifier_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Metric.h"
#include "N3L.h"

struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 256;

public:
	vector<SparseNode> output;

public:
	ComputionGraph() : Graph(){
		createNodes(max_sentence_length);
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length){
		output.resize(sent_length);
	}

	inline void clear(){
		Graph::clear();
		output.clear();
	}

public:
	inline void initial(SparseParams& olayer_linear){
		for (int idx = 0; idx < max_sentence_length; idx++) {
			output[idx].setParam(&olayer_linear);
		}
	}

public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const vector<Feature>& features, bool bTrain = false){
		clearValue(); // compute is a must step for train, predict and cost computation
		int seq_size = features.size();

		for (int idx = 0; idx < seq_size; idx++) {
			const Feature& feature = features[idx];
			output[idx].forward(feature.linear_features);
			execs.push_back(&output[idx]);
		}

	}
};

//A native neural network classfier using only word embeddings
class SparseClassifier {
public:
	SparseClassifier() {
		_dropOut = 0.0;
		_pcg = NULL;
	}
	~SparseClassifier() {
		if (_pcg != NULL)
			delete _pcg;
		_pcg = NULL;
	}

public:
	int _labelSize;

	dtype _dropOut;
	Metric _eval;

	SparseParams _layer_linear;
	SoftMaxLoss _loss;

	ModelUpdate _ada;

	ComputionGraph *_pcg;

	CheckGrad _checkgrad;

public:

	inline void init() {
		if (_layer_linear.nVSize == 0 || _layer_linear.nDim == 0){
			std::cout << "Please initialize linear features before this" << std::endl;
			return;
		}

		_labelSize = _layer_linear.nDim;

		_eval.reset();

		_layer_linear.exportAdaParams(_ada);

		_pcg = new ComputionGraph();
		_pcg->initial(_layer_linear);

		_checkgrad.add(&(_layer_linear.W), "_layer_linear.W");

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
			_loss.predict(&(_pcg->output[idx]), results[idx]);
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

#endif /* SRC_SparseClassifier_H_ */
