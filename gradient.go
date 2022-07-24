package goneural

type GradientFunc func(layersIterator, [][]float64, [][]float64) float64

func BatchGradient(li layersIterator, x, y [][]float64) float64 {
	return 0
}
