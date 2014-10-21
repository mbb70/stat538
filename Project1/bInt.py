import sys
import numpy as np
import scipy.integrate as integ
import scipy.stats as stats

def make_brownian_sigma(t):
    size = len(t)
    sigma = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            sigma[i][j] = min(t[i], t[j])
    return sigma

def make_ranges(n, lower_limits, upper_limits):
    ranges = np.zeros((n, 2))
    for i in range(n):
        ranges[i] = [lower_limits[i], upper_limits[i]]
    return ranges

def multivariate_gaussian_integration(sigma, lower_limits, upper_limits, mu):
    n = len(sigma)
    inv_sigma = np.linalg.inv(sigma)
    det_sigma = np.linalg.det(sigma)
    coeff_denominator = (2*np.pi)**(n/2.0)*np.sqrt(abs(det_sigma))
    coeff = 1.0/coeff_denominator
    f = lambda *x : np.exp(-0.5*np.dot(np.dot((np.array(x)-mu), inv_sigma), (np.array(x)-mu).T))
    ranges = make_ranges(n, lower_limits, upper_limits)
    return coeff*integ.nquad(f, ranges)[0]

def main(times):
    t = np.array([1,2,3])
    sigma = make_brownian_sigma(t)

    #print 'Brownian motion without drift'
    drift = 0
    mu = t*drift
    lower_limits, upper_limits = [-1,-1,-1], [1,1,1]
    #print multivariate_gaussian_integration(sigma, lower_limits, upper_limits, mu)
    tot = 0
    for i in range(times):
        tot += stats.mvn.mvnun(lower_limits, upper_limits, mu, sigma)[0]

"""
    lower_limits, upper_limits = [-1,-2,-3], [1,2,3]
    #print 'P[-1<B(1)<1, -2<B(2)<2, -3<B(3)<3]', multivariate_gaussian_integration(sigma, lower_limits, upper_limits, mu)
    print 'P[-1<B(1)<1, -2<B(2)<2, -3<B(3)<3]', stats.mvn.mvnun(lower_limits, upper_limits, mu, sigma)[0]

    print '\nBrownian motion with drift 1'
    drift = 1
    mu = t*drift
    lower_limits, upper_limits = [-1,-1,-1], [1,1,1]
    #print 'P[-1<B(1)<1, -1<B(2)<1, -1<B(3)<1]', multivariate_gaussian_integration(sigma, lower_limits, upper_limits, mu)
    print 'P[-1<B(1)<1, -1<B(2)<1, -1<B(3)<1]', stats.mvn.mvnun(lower_limits, upper_limits, mu, sigma)[0]

    lower_limits, upper_limits = [-1,-2,-3], [1,2,3]
    #print 'P[-1<B(1)<1, -2<B(2)<2, -3<B(3)<3]', multivariate_gaussian_integration(sigma, lower_limits, upper_limits, mu)
    print 'P[-1<B(1)<1, -2<B(2)<2, -3<B(3)<3]', stats.mvn.mvnun(lower_limits, upper_limits, mu, sigma)[0]
""" 
if __name__ == "__main__":
    main(int(sys.argv[1]))