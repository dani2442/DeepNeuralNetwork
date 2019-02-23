#pragma once

#include <Eigen/Core>
#include <vector>
#include <stdexcept>
#include "../Layer.h"
#include "../Config.h"
#include "../Utils/MaxAverage.h"

template<typename Activation>
class Pooling : public Layer {
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::MatrixXi IntMatrix;

	const int m_channel_rows;
	const int m_channel_cols;
	const int m_in_channel;
	const int m_pool_rows;
	const int m_pool_cols;
	const int m_out_rows;
	const int m_out_cols;

	IntMatrix m_loc;
	Matrix m_z;
	Matrix m_a;
	Matrix m_din;

public:
	Pooling(const int in_width,const int in_height,const int in_channel,const int pooling_width,const pooling_height):
		Layer(in_width*in_height*in_channel,(in_width/pooling_width)*(in_height/pooling_height)*in_channel),
		m_channel_rows(in_height),
		m_channel_cols(in_width),
		m_in_channel(in_channel),
		m_pool_rows(pooling_height),
		m_pool_cols(pooling_width),
		m_out_rows(m_channel_rows/m_pool_rows),
		m_out_cols(m_channel_cols/m_pool_cols)
	{}

	void init(const Scalar& mu,const Scalar& sigma,RNG& rng){}

	void forward(const Matrix& prev_layer_data) {
		const int nobs = prev_layer_data.cols();
		m_loc.resize(this->m_out_size, nobs);
		m_z.resize(this->m_out_size, nobs);

		int* loc_data = m_loc.data();
		const int channel_end = prev_layer_data.size();
		const int channel_stride = m_channel_rows * m_channel_cols;
		const int col_end_gap = m_channel_rows * m_pool_cols*m_out_cols;
		const int col_stride = m_channel_rows * m_pool_cols;
		const row_end_gap = m_out_rows * m_pool_rows;
		for (int channen_start = 0; channel_start < channel_end; channen_start += channel_stride) {
			const int col_end = channen_start + col_end_gap;
			for (int col_start = channel_start : col_start < col_end; col_start += col_stride) {
				const int row_end = col_start + row_end_gap;
				for (int row_start = col_start; row_start < row_end; rows_start += m_pool_rows, loc_data++) {
					loc_data = row_start;
				}
			}
		}

		loc_data = m_loc.data();
		const int* const loc_end = loc_data + m_loc.size();
		Scalar* z_data = m_z.data();
		const Scalar* src = prev_layer_data.data();
		for (; loc_data < loc_end; loc_data++, z_data++) {
			const int offset = *loc_data;
			*z_data = internal::find_block_max(src + offset, m_pool_rows, m_pool_cols, m_channel_rows, *loc_data);
			*loc_data += offset;
		}

		m_a.resize(this->m_out_size, nobs);
		Activation::activate(m_z, m_a);
	}

	const Matrix& output() const { return m_a; }

	void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data) {
		// TODO
	}

	const Matrix& backprop_data() const { return m_din; }

	void update(Optimizer& opt){}
};