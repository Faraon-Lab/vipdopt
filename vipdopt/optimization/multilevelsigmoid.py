import numpy as np
import matplotlib.pyplot as plt

# Sigmoid parameters
eta = 0.5
beta = 0.0625*2**7      # can be any value

alpha = 1 # 1/3         # only can be values [0,1]

# Vectors for plotting
x = np.linspace(-0.5,5.0, 1000)

ylims = np.array([1.6,2.1,2.5,3.2])
xlims = np.copy(ylims)
ylims_0 = np.copy(ylims)
xlims_0 = np.copy(xlims)
print(f'Mean is {np.mean(ylims)}')
ylims = ylims + alpha*(np.mean(ylims) - ylims)
num_sigmoids = len(ylims)-1

# Calculate denominator for use in methods
denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
numerators = np.zeros((num_sigmoids, len(x)))
funcs = np.zeros((num_sigmoids, len(x)))
for i in range(num_sigmoids):
    numerators[i] = np.tanh(beta * eta) + \
                    np.tanh( beta * 1/(xlims[i+1]-xlims[i]) * (
                        ( x - np.mean([xlims[i],xlims[i+1]]) ) 
                    ) )
    funcs[i] = ylims[i] + (ylims[i+1]-ylims[i]) * (numerators[i] / denominator)
    plt.plot(x, funcs[i], '-', label=f'f{i}')
    
plt.plot(x,x,'k--')
# plt.plot(x, (np.tanh(beta*eta)+np.tanh(beta*(x-eta)))/denominator)
# plt.plot(x, (np.tanh(beta*eta)+np.tanh(beta*1/10*(x-eta)))/denominator)
plt.vlines(xlims_0,0,5,color='gray', alpha=0.4)
plt.hlines(ylims_0,-1.5,5,color='gray', alpha=0.5)
plt.xlim((xlims_0[0]-eta,xlims_0[-1]+eta))
plt.ylim((ylims_0[0]-eta,ylims_0[-1]+eta))
plt.legend()
plt.show()

print('End of code reached.')