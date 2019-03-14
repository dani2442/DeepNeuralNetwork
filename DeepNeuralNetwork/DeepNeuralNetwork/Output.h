#pragma once

#include <Eigen/Core>
#include <stdexcept>
#include "Config.h"

class Output {
protected:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1>Vector;
	typedef Eigen::RowVectorXi IntegerVector;

public:
	virtual ~Output() {	}

	virtual void evaluate(const Matrix& prev_layer_data, const Matrix& target) = 0;
	virtual void evaluate(const Matrix& prev_layer_data, const IntegerVector& target) = 0;

	virtual const Matrix& backprop_data()const = 0;
	virtual Scalar loss() const = 0;
};