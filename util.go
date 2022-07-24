package goneural

func sameSize(slc []float64, slcs ...[]float64) bool {
	l := len(slc)
	for _, s := range slcs {
		if len(s) != l {
			return false
		}
	}
	return true
}
