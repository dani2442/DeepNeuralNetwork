#pragma once
#include "Config.h"

namespace internal {

	inline int find_max(const Scalar* x, const int n) {
		int loc = 0;
		for(int i = 1; i < n; i++)
		{
			loc = (x[i] > x[loc]) ? i : loc;
		}
		return loc;
	}

	inline Scalar find_block_max(const Scalar* x, const int nrow, const int ncol, const int col_stride, int& loc)
	{
		// Max element in the first column
		loc = find_max(x, nrow);
		Scalar val = x[loc];

		x += col_stride;
		int loc_next = find_max(x, nrow);
		Scalar val_next = x[loc_next];
		if(val_next > val)  { loc = col_stride + loc_next; val = val_next; }
		if(ncol == 2)  return val;

		// Other columns
		for(int i = 2; i < ncol; i++)
		{
			x += col_stride;
			loc_next = find_max(x, nrow);
			val_next = x[loc_next];
			if(val_next > val)  { loc = i * col_stride + loc_next; val = val_next; }
		}

		return val;
	}

	inline Scalar sum_row(const Scalar* x, const int n)
	{
		Scalar c = 0;
		for(int i = 0; i < n; i++)
		{
			c += x[i];
		}
		return c;
	}

	inline Scalar average_block(const Scalar* x, const int nrow, const int ncol, const int col_stride, int& loc)
	{
		Scalar sum = 0;
		for(int i = 0; i < ncol; i++)
		{
			x += col_stride;
			sum += sum_row(x, nrow);
		}
		return sum / (ncol*nrow);
	}
}