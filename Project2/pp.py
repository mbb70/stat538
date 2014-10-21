import matplotlib.pyplot as plt
import numpy as np


mu, num_jumps, num_times = 1, 10+1, 10
for i in range(num_times):
    time_jumps = np.random.exponential(1.0/mu, num_jumps)
    wt = np.cumsum(time_jumps)
    plt.step(wt, range(num_jumps), lw=2)

plt.rc('text', usetex=True)
plt.title(r'\huge %s Poisson Processes with \boldmath{$\mu=%s$}' % (num_times, mu))
plt.xlabel(r'\huge time')
plt.ylabel(r'\huge $\mathbf{N(t)}$')
plt.tick_params(axis='both', labelsize=16)
plt.show()
