
import seaborn as sns
import numpy as np
import pandas as pd

def ppf(x,adj):
    if x == 0:
        x = .00000000001
    ppf = (1.-adj*np.log10(x)) / (adj * 10.**(1/adj))
    return ppf

adjValues = [.5,.7,1.,1.5,2.,3.,5.,10.]
intercepts = [10**(1/adj) for adj in adjValues]
maxIntercept = max(intercepts)

df = pd.DataFrame(columns=adjValues)
for x in np.linspace(0,50.,100):
    df.loc[x,:] = list(map(ppf,[x]*len(adjValues),adjValues))



sns.scatterplot(data=df)
