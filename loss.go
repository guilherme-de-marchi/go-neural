package goneural

import (
	"errors"
	"math"
)

const epsilon = 1e-15

type LossFunc func([]float64, []float64) (float64, error)

func MeanAbsoluteError(y, yp []float64) (float64, error) {
	if !sameSize(y, yp) {
		return 0, errors.New("slices with different sizes.")
	}

	var totalError float64
	for i, v := range y {
		totalError += math.Abs(v - yp[i])
	}
	return totalError / float64(len(y)), nil
}

func MeanSquaredError(y, yp []float64) (float64, error) {
	if !sameSize(y, yp) {
		return 0, errors.New("slices with different sizes.")
	}

	var totalError float64
	for i, v := range y {
		totalError += math.Pow(v-yp[i], 2)
	}
	return totalError / float64(len(y)), nil
}

// Replace 0 by epsilon, and
// 1 by 1-epsilon, so a binary cross entropy
// can be run without issues.
func prepareToBinaryCrossEntropy(slc []float64) []float64 {
	for i, v := range slc {
		switch v {
		case 0:
			slc[i] = epsilon
		case 1:
			slc[i] = 1 - epsilon
		}
	}
	return slc
}

func BinaryCrossEntropy(y, yp []float64) (float64, error) {
	if !sameSize(y, yp) {
		return 0, errors.New("slices with different sizes.")
	}

	yp = prepareToBinaryCrossEntropy(yp)
	var totalError float64
	var w float64
	for i, v := range y {
		w = yp[i]
		totalError += v*math.Log(w) + (1-v)*math.Log(1-w)
	}
	return -(totalError / float64(len(y))), nil
}
