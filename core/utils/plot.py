import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import LinearLocator
from matplotlib import pyplot as plt
import numpy as np


def xy_curves(x, y):
    fig, ax, = plt.subplots(1, figsize=(7, 7))
    ax.get_yaxis().set_major_locator(LinearLocator(numticks=12))
    for i in range(len(x)):
        color = np.random.uniform(0, 1, 3).tolist()
        ax.semilogx(x[i], y[i], color=color, label=str(i))
    ax.legend()
    ax.grid(True)
    return fig

def bar(x):
    fig2, ax2, = plt.subplots(1, figsize=(7, 7))
    ax2.bar(np.arange(len(x)), x, align='center', alpha=0.5,
           color='red',
           edgecolor='blue')
    plt.xticks(np.arange(len(x)), [str(i) for i in range(len(x))])
    plt.ylabel('AP')
    return fig2

if __name__ == '__main__':
    x = [np.linspace(0, 1, 100) for i in range(10)]
    y = [item**np.random.rand() for item in x]


    x = np.random.uniform(0, 1, 10)
    fig = bar(x)
    plt.show()
