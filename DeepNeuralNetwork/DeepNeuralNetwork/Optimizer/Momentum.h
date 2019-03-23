#pragma once
#include <Eigen/Core>
#include "../Utils/sparsepp.h"
#include "../Config.h"
#include "../Optimizer.h"

class Momentum :public Optimizer {
private:
	typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;
	spp::sparse_hash_map<const Scalar*, Array> m_history;

public:
	Scalar m_lrate;
	Scalar m_momentum;
	Scalar m_decay;

	Momentum(): m_lrate(Scalar(0.01)),m_momentum(Scalar(0.9)),m_decay(Scalar(0)){}

	void reset() { m_history.clear(); }

	void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) {
		Array& grad = m_history[dvec.data()];

		if (grad.size() == 0) {
			grad.resize(dvec.size());
			grad.setZero();
		}

		grad = grad * m_momentum + m_lrate * (dvec + m_decay * vec).array();

		vec.array() -= grad;
	}

};