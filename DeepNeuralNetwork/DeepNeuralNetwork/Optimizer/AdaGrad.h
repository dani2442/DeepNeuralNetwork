#pragma once
#include <Eigen/Core>
#include "../Config.h"
#include "../Optimizer.h"
#include "../Utils/sparsepp.h"

class AdaGrad :public Optimizer {
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1>Vector;
	typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;

	spp::sparse_hash_map<const Scalar*, Array> m_history;

public:
	Scalar m_lrate;
	Scalar m_eps;

	AdaGrad():m_lrate(Scalar(0.01)),m_eps(Scalar(1e-7)){}

	void reset() { m_history.clear(); }

	void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) {
		Array& grad_square = m_history[dvec.data()];
		if (grad_square.size() == 0) {
			grad_square.resize(dvec.size());
			grad_square.setZero();
		}
		grad_square += dvec.array().square();

		vec.array() -= m_lrate * dvec.array() / (grad_square.sqrt() + m_eps);
	}
};