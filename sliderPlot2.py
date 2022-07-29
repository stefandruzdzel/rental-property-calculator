
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import betaprime
import pandas as pd
import matplotlib.ticker as mtick
from io import BytesIO
import base64

def Main(alpha,targetMean,skew):


    X_AXIS_CONSTANT = 5

    # The parametrized function to be plotted
    def distribution(t, beta, targetMean):
        #Squared so it scales faster, making sure it's not exactly 1 because that errors
        #beta = (beta+.05)**1.4
        beta = 1.2 ** beta - .15
        mean, var, skew, kurt = betaprime.stats(alpha, beta, moments='mvsk')
        scale = targetMean/mean
        cdf = betaprime.cdf(t/scale, alpha, beta)
        hist = betaprime.rvs(alpha, beta, size=2000)*scale
        return cdf,hist

    t = np.linspace(0, targetMean * X_AXIS_CONSTANT, 1000)

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()

    # Format the Y-axis as percentage
    yfmt = '%.0f%%' 
    yticks = mtick.FormatStrFormatter(yfmt)
    ax.yaxis.set_major_formatter(yticks)

    xfmt = '$'+'%.0f'
    xticks = mtick.FormatStrFormatter(xfmt)
    ax.xaxis.set_major_formatter(lambda x,pos:'$'+str(int(x)))

    ax.set_xlabel('Monthly maintenance')


    cdf,hist = distribution(t, skew, targetMean)

    pd.DataFrame(hist,columns=['Likelihood']).plot(ax=ax,kind='hist',weights=np.ones(2000)*100./2000,bins=range(0,int(targetMean * X_AXIS_CONSTANT)+100,100))
    pd.DataFrame(cdf*100.,index=t,columns=['Cumulative Likelihood (CDF)']).plot(ax=ax,kind='line')

    ax.set_xlim(0, targetMean * X_AXIS_CONSTANT)
    ax.set_ylim(0,100)
    fig.canvas.draw_idle()

    # Create the html for the png file
    #tmpfile = BytesIO()
    #fig.savefig(tmpfile, format='png')
    #encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    #html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

    #plt.show()
    #return html
    return fig

# x = Main(plotType='betaPrime1',targetMean=200,skew=3)
# print(x)