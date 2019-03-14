#pragma once

#include <Eigen/Core>
#include "../Config.h"

class MSE :public Output {
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

	Matrix m_din;

public:
	void evaluate(const Matrix& prev_layer_data, const Matrix& target) {
		const int nobs = prev_layer_data.cols();
		const int nvar = prev_layer_data.rows();

		// Compute the derivative of the input layer
		m_din.resize(nvar, nobs);
		m_din.noalias() = prev_layer_data - target;
	}

	const Matrix& backprop_data() const { return m_din; }

	Scalar loss() const {
		return m_din.squaredNorm() / m_din.cols()*0.5;
	}
};