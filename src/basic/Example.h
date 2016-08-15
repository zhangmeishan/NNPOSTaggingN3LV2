/*
 * Example.h
 *
 *  Created on: Mar 17, 2015
 *      Author: mszhang
 */

#ifndef SRC_EXAMPLE_H_
#define SRC_EXAMPLE_H_

#include <vector>
#include <string>

using namespace std;
struct Feature {
public:
	vector<string> words;
	vector<string> chars;
	vector<string> linear_features;
public:
	Feature() {
	}

	//virtual ~Feature() {
	//
	//}

	void clear() {
		words.clear();
		chars.clear();
		linear_features.clear();
	}
};

class Example {

public:
  vector<vector<double> > m_labels;
  vector<Feature> m_features;

public:
  Example()
  {

  }
  virtual ~Example()
  {

  }

  void clear()
  {
    m_labels.clear();
    m_features.clear();
  }


};


#endif /* SRC_EXAMPLE_H_ */
