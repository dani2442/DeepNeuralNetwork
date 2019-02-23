#pragma once
#include <Eigen/Core>
#include "../Config.h"

class Softmax
{
private:
	typedef Eigen::Array<Scalar, 1, Eigen::Dynamic>RowArray;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:
	static inline void activate(const Matrix& Z, Matrix& A) {
		A.array() = (Z.rowwise() - Z.colwise().maxCoeff()).array().exp();
		RowArray colsum = A.colwise().sum();
		A.array().rowwise() /= colsum;
	}

	static inline void apply_jacobian(const Matrix& Z , const Matrix& A, const Matrix& F, Matrix& G) {
		RowArray a_dot_f = A.cwiseProduct(F).colwise().sum();
		G.array() = A.array()*(F.array().rowwise() - a_dot_f);
	}
};