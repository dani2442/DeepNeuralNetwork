#pragma once
#include <Eigen/Core>
#include "../Optimizer.h"
#include "../Config.h"

class SGD : public Optimizer {
public:
	Scalar m_lrate;
	Scalar m_decay;

	SGD(): m_lrate(Scalar(0.001)),m_decay(Scalar(0)){}

	void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) {
		vec.noalias() -= m_lrate * (dvec + m_decay * vec);
	}
};