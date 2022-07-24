package main

import (
	"fmt"

	"github.com/Guilherme-De-Marchi/goneural"
)

func main() {
	model := goneural.NewModel(goneural.BinaryCrossEntropy)
	err := model.AddLayer(&goneural.Layer{
		Inputs:     2,
		Outputs:    3,
		Activation: goneural.Sigmoid,
	})
	if err != nil {
		panic(err)
	}
	err = model.AddLayer(&goneural.Layer{
		Outputs:    1,
		Activation: goneural.Sigmoid,
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(model.GetLayers())
	err = model.Fit(
		[][]float64{
			{.22, 1},
			{.29, 0},
			{.55, 0},
			{.58, 1},
		},
		[][]float64{
			{0},
			{0},
			{0},
			{1},
		},
		2,
	)
	if err != nil {
		panic(err)
	}
}
