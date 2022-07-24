package goneural

import (
	"errors"

	"github.com/Guilherme-De-Marchi/goneural/matrix"
)

type Layer struct {
	Inputs, Outputs int
	weights         [][]float64
	bias            []float64
	Activation      ActivationFunc
}

func (l *Layer) GenerateWeights(f func() float64) {
	l.weights = make([][]float64, l.Outputs)
	for i := range l.weights {
		l.weights[i] = make([]float64, l.Inputs)
		for j := range l.weights[i] {
			l.weights[i][j] = f()
		}
	}
}

func (l *Layer) GenerateBias(f func() float64) {
	l.bias = make([]float64, l.Outputs)
	for i := range l.weights {
		l.bias[i] = f()
	}
}

func (l *Layer) Forward(x []float64) ([]float64, error) {
	if len(x) != l.Inputs {
		return nil, errors.New("input array lenght is different from the lenght expected.")
	}

	dot, err := matrix.DotProduct(x, l.weights)
	if err != nil {
		return nil, err
	}
	sum, err := matrix.VecSum(dot, l.bias)
	if err != nil {
		return nil, err
	}
	return matrix.VecApplyFunc(sum, l.Activation), nil
}

type layersIterator struct {
	layers  []*Layer
	Current *Layer
	currId  int
}

func newLayersIterator(layers []*Layer) *layersIterator {
	return &layersIterator{layers: layers}
}

func (li *layersIterator) Next() bool {
	if li.currId == len(li.layers) {
		li.Current = nil
		return false
	}
	li.Current = li.layers[li.currId]
	li.currId += 1
	return true
}
