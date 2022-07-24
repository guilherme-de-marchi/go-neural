package matrix

import (
	"errors"
)

func DotProduct(a []float64, b [][]float64) ([]float64, error) {
	c := make([]float64, len(b))
	for i, v := range b {
		if len(v) != len(a) {
			return nil, errors.New("lenght of the B rows and A must be the same")
		}

		var x float64
		for j, w := range v {
			x += w * a[j]
		}
		c[i] = x
	}

	return c, nil
}

func VecSum(a, b []float64) ([]float64, error) {
	if len(a) != len(b) {
		return nil, errors.New("lenght of A and B must be the same")
	}

	c := make([]float64, len(a))
	for i, v := range a {
		c[i] = v + b[i]
	}
	return c, nil
}

func VecApplyFunc(a []float64, f func(float64) float64) []float64 {
	b := make([]float64, len(a))
	for i, v := range a {
		b[i] = f(v)
	}
	return b
}
