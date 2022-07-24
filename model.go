package goneural

import (
	"errors"
	"fmt"
)

type model struct {
	layers []*Layer
	loss   LossFunc
}

func NewModel(loss LossFunc) *model {
	return &model{loss: loss}
}

func (m *model) Fit(xTrain, yTrain [][]float64, epochs int) error {
	if len(xTrain) != len(yTrain) {
		return errors.New("train arrays have different sizes.")
	}
	if epochs == 0 {
		return errors.New("epochs must be higher than 0.")
	}

	yPredicted := make([][]float64, len(xTrain))
	var totalError float64
	for epoch := 0; epoch < epochs; epoch++ {
		var err error
		for i, set := range xTrain {
			li := newLayersIterator(m.layers)
			currSlc := set
			for li.Next() {
				currSlc, err = li.Current.Forward(currSlc)
				if err != nil {
					return err
				}
			}
			yPredicted[i] = currSlc

			if len(yTrain[i]) != len(yPredicted[i]) {
				return errors.New("yTrain and yPredicted arrays have different sizes.")
			}

			localError, err := m.loss(yTrain[i], yPredicted[i])
			if err != nil {
				return err
			}
			totalError += localError
		}

		// Regression here
	}

	fmt.Println(yPredicted, totalError)
	return nil
}

// Adds a new fully connected layer on the
// layers chain.
// This method will overwrite the field Layer.Inputs
// of all layers added on a model, except the first,
// by the value on Layer.Outputs of the previous layer,
// and generate weights and bias for each layer.
func (m *model) AddLayer(l *Layer) error {
	if l.Activation == nil {
		return errors.New("missing field Activation.")
	}
	if l.Outputs == 0 {
		return errors.New("field Outputs can not be 0.")
	}

	if len(m.layers) == 0 {
		if l.Inputs == 0 {
			return errors.New("field Inputs can not be 0.")
		}
	} else {
		l.Inputs = m.layers[len(m.layers)-1].Outputs
	}

	// the user should be able to choose the generation method.
	l.GenerateWeights(func() float64 { return 1 })
	l.GenerateBias(func() float64 { return 0 })

	m.layers = append(m.layers, l)
	return nil
}

func (m *model) GetLayers() []Layer {
	layers := make([]Layer, len(m.layers))
	for i, v := range m.layers {
		layers[i] = *v
	}
	return layers
}
