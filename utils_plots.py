
import matplotlib.pyplot as plt
import numpy as np

def hist_1d(x,bins=50,show=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    num, bins_, patches = ax.hist(x, bins, normed=1)
    plt.show()


def hist_2d(x,y,bins=50,show=False):
    heatmap, xedges, yedges = np.histogram2d(x,y,bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.figure()
    plt.clf()
    plt.imshow(heatmap.T,extent=extent,origin='lower')
    plt.show()

def line_2d(x,y,linetype='-',ylims=None):
    if y.ndim==1:
        y = np.reshape(y,(len(y),1))
    if x.ndim==1:
        x = np.reshape(x,(len(x),1))
    plt.figure()
    for d in range(np.shape(y)[1]):
        id_x = min(d,x.shape[1]-1)
        plt.plot( x[:,id_x], y[:,d], linetype )

    if ylims!=None:
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,ylims[0],max(ylims[1],y2)))
    plt.show()

