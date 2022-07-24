package goneural

import "math"

type ActivationFunc func(float64) float64

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
