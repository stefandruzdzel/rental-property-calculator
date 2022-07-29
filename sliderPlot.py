
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.stats import betaprime
import pandas as pd
import matplotlib.ticker as mtick

def Main():
    X_AXIS_CONSTANT = 5

    # The parametrized function to be plotted
    def distribution(t, beta, targetMean):
        alpha = 1
        #Squared so it scales faster, making sure it's not exactly 1 because that errors
        #beta = (beta+.05)**1.4
        beta = 1.2 ** beta - .15
        mean, var, skew, kurt = betaprime.stats(alpha, beta, moments='mvsk')
        scale = targetMean/mean
        cdf = betaprime.cdf(t/scale, alpha, beta)
        hist = betaprime.rvs(alpha, beta, size=2000)*scale
        return cdf,hist



    # Define initial parameters
    init_targetMean = 200
    init_skew = 2

    t = np.linspace(0, init_targetMean * X_AXIS_CONSTANT, 1000)

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()

    # Format the Y-axis as percentage
    yfmt = '%.0f%%' 
    yticks = mtick.FormatStrFormatter(yfmt)
    #ax.yaxis.set_major_formatter(yticks)

    xfmt = '$'+'%.0f'
    xticks = mtick.FormatStrFormatter(xfmt)
    #ax.xaxis.set_major_formatter(xticks)


    #cdf,hist = distribution(t, init_skew, init_targetMean)
    #pd.DataFrame(hist,columns=['Likelihood']).plot(ax=ax,kind='hist',weights=np.ones(2000)*100./2000,bins=range(0,int(init_targetMean * X_AXIS_CONSTANT)+100,100))
    #pd.DataFrame(cdf*100.,index=t,columns=['Cumulative Likelihood (CDF)']).plot(ax=ax,kind='line')


    #ax.set_xlim(0, init_targetMean * X_AXIS_CONSTANT)

    ax.set_xlabel('Monthly maintenance')

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control the skew.
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    skew_slider = Slider(
        ax=axfreq,
        label='Distribution Skew',
        valmin=1,
        valmax=10,
        valinit=init_skew,
    )

    # Make a vertically oriented slider to control the targetMean
    axamp = plt.axes([0.1, 0.25, 0.0225, 0.63])
    mean_slider = Slider(
        ax=axamp,
        label="Avg Mo. Maintenance",
        valmin=0,
        valmax=1000,
        valinit=init_targetMean,
        orientation="vertical",
        valfmt="$"+'%.0f')


    # The function to be called anytime a slider's value changes
    def update(val):
        ax.cla()
        ax.yaxis.set_major_formatter(yticks)
        ax.xaxis.set_major_formatter(lambda x,pos:'$'+str(int(x)))
        t = np.linspace(0, mean_slider.val * X_AXIS_CONSTANT, 1000)

        cdf,hist = distribution(t, skew_slider.val, mean_slider.val)

        pd.DataFrame(hist,columns=['Likelihood']).plot(ax=ax,kind='hist',weights=np.ones(2000)*100./2000,bins=range(0,int(mean_slider.val * X_AXIS_CONSTANT)+100,100))
        pd.DataFrame(cdf*100.,index=t,columns=['Cumulative Likelihood (CDF)']).plot(ax=ax,kind='line')


        ax.set_xlim(0, mean_slider.val * X_AXIS_CONSTANT)
        ax.set_ylim(0,100)
        fig.canvas.draw_idle()


    # register the update function with each slider
    skew_slider.on_changed(update)
    mean_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')


    def reset(event):
        skew_slider.reset()
        mean_slider.reset()
    button.on_clicked(reset)

    update(0)
    


    from io import BytesIO
    import base64
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
    print(html)


    plt.show()
    return html

x = Main()
print(x)