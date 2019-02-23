#pragma once
#include <Eigen/Core>
#include <vector>
#include <stdexcept>
#include "../Config.h"
#include "../Layer.h"
#include "../Utils/Convolution.h"
#include "../Utils/Random.h"

template<typename Activation>
class Convolutional :public Layer {
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Matrix::ConstAlignedMapType ConstAlignedMapMat;
    typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
    typedef Vector::AlignedMapType AlignedMapVec;

	const internal::ConvDims m_dim;

	Vector m_filter_data;
	Vector m_df_data;

	Vector m_bias;
	Vector m_db;

	Matrix m_z;
	Matrix m_a;
	Matrix m_din;

public:
	Convolutional(const int in_width, const int in_height, const int in_channels, const int out_channels, const int window_width, const int window_height) :
		Layer(in_width*in_height*in_channels, (in_width - window_width + 1)*(in_height - window_height + 1)*out_channels),
		m_dim(in_channels, out_channels, in_height, in_width, window_height, window_width)
	{}

	void init(const Scalar&mu, const Scalar& sigma, RNG& rng) {
		const int filter_data_size = m_dim.in_channels*m_dim.out_channels*m_dim.filter_rows*m_dim.filter_cols;
		m_filter_data.resize(filter_data_size);
		m_df_data.resize(filter_data_size);

		m_bias.resize(m_dim.out_channels);
		m_db.resize(m_dim.out_channels);

		internal::set_normal_random(m_filter_data.data(), filter_data_size, rng, mu, sigma);
		internal::set_normal_random(m_bias.data(), m_dim.out_channels, rng, mu, sigma);
	}

	void forward(const Matrix& prev_layer_data) {
		const int nobs = prev_layer_data.cols();
		m_z.resize(this->m_out_size, nobs);

		internal::convolve_valid(m_dim, prev_layer_data.data(), true, nobs, m_filter_data.data(), m_z.data());

		int channel_start_row = 0;
		const int channel_nelem = m_dim.conv_rows*m_dim.conv_cols;
		for (int i = 0; i < m_dim.out_channels; i++, channel_start_row += channel_nelem) {
			m_z.block(channel_start_row, 0, channel_nelem, nobs).array() += m_bias[i];
		}

		m_a.resize(this->m_out_size, nobs);
		Activation::activate(m_z, m_a);
	}

	const Matrix& output() const {
		return m_a;
	}

	void backprop(const Matrix& pre_layer_data, const Matrix& next_layer_data) {

	}

	const Matrix& backprop_data() const {
		return m_din;
	}

	void update(Optimizer& opt) {
		ConstAlignedMapVec dw(m_df_data.data(), m_df_data.size());
		ConstAlignedMapVec db(m_db.data(), m_db.size());
		AlignedMapVec w(m_filter_data.data(), m_df_data.size());
		AlignedMapVec b(m_db.data(), m_bias.size());

		opt.update(dw, w);
		opt.update(db, b);
	}
};
