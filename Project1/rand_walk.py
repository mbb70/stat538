import numpy as np
import matplotlib.pyplot as plt

def make_walk(prob, size):
    rand_vals = 2*np.random.binomial(1, prob, size) - 1
    rand_walk = np.zeros(size)
    for i in range(size):
        rand_walk[i] = rand_walk[i-1] + rand_vals[i]
    return rand_walk

def main():
	size = 100000
	x = np.asfortranarray(range(size))
	for times in range(10):
	    plt.plot(x, make_walk(0.5, size))
	plt.title('Ten 100,000 Step Unbiased Random Walks')
	plt.show()

	plt.figure(2)
	plt.plot(x, make_walk(0.51, size), 'k-', label='p = 0.51')
	plt.plot(x, make_walk(0.49, size), 'b-', label='p = 0.49')
	for prob, color in [[0.51, 'k-'], [0.49, 'b-']]:
	    for times in range(4):
	        plt.plot(x, make_walk(prob, size), color)
	plt.legend(loc='upper left')
	plt.title('Ten 100,000 Step Biased Random Walks')
	plt.show()

if __name__ == '__main__':
	main()