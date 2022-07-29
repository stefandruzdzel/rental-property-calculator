
import pickle
import pandas as pd
from scipy.stats import norm
import numpy as np
from scipy.stats import betaprime
import datetime as dt
import matplotlib.ticker as mtick
import base64
from io import BytesIO


# Interest Rates
# Note, All interest rates are done in whole numbers, IE: 5 = 5%, NOT .05
INTEREST_RATE_STD = 1.92
ARM_MINIMUM_RATE = 2.
def usd(x):
    # Format value as USD
    return ["${:,.2f}".format(value).replace('$-','-$') for value in x]

def percent(x):
    # Format value as Percent. .101 = '10.1%'
    return ["{:.1%}".format(value) for value in x]

def generateNormalDistribution(mean,std,multiple):    
    array30 = (mean + norm.rvs(size=30)*std)/100.+1.
    array30 = np.cumproduct(array30)
    
    array360 = np.array([np.nan]*360)
    array360[range(11,360,12)] = array30
    array360[0] = 1 + (array30[0]-1)/12 #Add a month of appreciation
    array360 = pd.Series(array360).interpolate().values*multiple
    return array360

def convertFigureToHTML(fig):
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
    return html

vars = pickle.load(open(r'C:\VS Code\rental-property-calculator\temp','rb'))

price, down, rate, term, closingCosts, meanAppreciation, appreciationStd, sellCost, armBool, armYears, armMargin, armAAC, armLAC, pppBool, pppYears, pppFee, tax, insurance, maintenance, maintenanceSkew, otherCosts, mean_inflation, std_inflation, rent, mean_rentGrowth,  std_rentGrowth, mean_stay, skew_stay, turnLength, pmBool, pmFeePerc, pmFeeFlat, pmLeaseFee = vars
pppFee = '321'
starterDF = pd.DataFrame(range(1,12*30+1),columns=['Month'])
starterDF.loc[:,'Year'] = starterDF.loc[:,'Month'].apply(lambda x:(x-1)//12+1)
starterDF.loc[:,'Rate'] = rate
starterDF.loc[:,['Prepayment Penalty','New Tenant','Turn Costs','Vacancy']] = 0.0, 0, 0.0, 0.0
loanAmount = price * (1 - down/100.)
loanLength = term * 12
initialCosts = price * down/100. + closingCosts

#Maintenance Beta Prime
maint_alpha = 1
maint_beta = 1.2 ** maintenanceSkew - .15
mean, _, _, _ = betaprime.stats(maint_alpha, maint_beta, moments='mvsk')
maint_scale = maintenance/mean

#Tenant Stay Beta Prime
stay_alpha = 10
stay_beta = 2.6+skew_stay/2.4
mean, _, _, _ = betaprime.stats(stay_alpha, stay_beta, moments='mvsk')
stay_scale = mean_stay/mean

global z1, z2, z3, z4, z5
z1 = dt.timedelta(0)
z2 = dt.timedelta(0)
z3 = dt.timedelta(0)
z4 = dt.timedelta(0)
z5 = dt.timedelta(0)

def singleIteration(x):
    t1 = dt.datetime.now()
    df = starterDF.copy()

    if armBool == 'Yes':
        previousRate = rate - armMargin
        for year in range(int(armYears)+1,31):
            bools = df.loc[:,'Year'] == year
            newRate = max(0, previousRate + norm.rvs()*INTEREST_RATE_STD)
            newRateLimited = min(min(armLAC+rate,newRate + armMargin),previousRate + armAAC + armMargin)
            df.loc[bools,'Rate'] = max(newRateLimited, ARM_MINIMUM_RATE)
            previousRate = newRate - armMargin
    global z1, z2, z3, z4, z5
    z1 += (dt.datetime.now() - t1)
    t1 = dt.datetime.now()
    
    
    
    if armBool == 'Yes':
        bools = df.loc[:,'Year'] <= armYears
    else:
        # Run these first calculations for the entire period if no ARM
        bools = df.loc[:,'Year'] <= term
    
    df.loc[bools,'Mortgage Payment'] = (loanAmount * (df.loc[bools,'Rate']/1200)) / (1 - (12/(12+df.loc[bools,'Rate']/100)) ** loanLength)
    df.loc[bools,'Balance'] = loanAmount * ((1 + df.loc[bools,'Rate']/1200) ** loanLength - (1 + df.loc[bools,'Rate']/1200) ** df.loc[bools,'Month']) / ((1 + df.loc[bools,'Rate']/1200) ** loanLength - 1)
    
    if armBool == 'Yes':
        for year in range(int(armYears)+1,int(term)+1):
              bools = df.loc[:,'Year'] == year
              remainingLength = (term - year + 1) * 12
              remainingBalance = df.loc[:,'Balance'].min()
              index = 12 * (year - 1)
              payment = (remainingBalance * (df.loc[index,'Rate']/1200)) / (1 - (12/(12+df.loc[index,'Rate']/100)) ** remainingLength)
              df.loc[bools,'Mortgage Payment']  = payment
              # df.loc[bools,'Mortgage Payment'] = (remainingBalance * (df.loc[bools,'Rate']/1200)) / (1 - (12/(12+df.loc[bools,'Rate']/100)) ** remainingLength)
              df.loc[bools,'Balance'] = remainingBalance * ((1 + df.loc[bools,'Rate']/1200) ** remainingLength - (1 + df.loc[bools,'Rate']/1200) ** (df.loc[bools,'Month']-((year - 1) * 12))) / ((1 + df.loc[bools,'Rate']/1200) ** remainingLength - 1)
    
    
    df.loc[0,'Principal'] = loanAmount - df.loc[0,'Balance']
    df.loc[df.loc[:,'Month'] > 1,'Principal'] = df.loc[df.loc[:,'Month'] < df.loc[:,'Month'].max(),'Balance'].values - df.loc[df.loc[:,'Month'] > 1,'Balance'].values 
    df.loc[:,'Interest'] = df.loc[:,'Mortgage Payment'] - df.loc[:,'Principal']
    df.loc[df.loc[:,'Year'] > term,['Mortgage Payment','Balance','Principal','Interest']] = 0.
        
    df.loc[:,'Home Value']  = generateNormalDistribution(meanAppreciation, appreciationStd, price)
    df.loc[:,'Market Rent'] = generateNormalDistribution(mean_rentGrowth, std_rentGrowth, rent)
    df.loc[:,'Inflation Index']   = generateNormalDistribution(mean_inflation, std_inflation, 1.)
    
    z2 += (dt.datetime.now() - t1)
    t1 = dt.datetime.now()
    #Prepayment Pentalty
    
    if pppBool == 'Yes':
        for i,char in enumerate(pppFee):
             bools = df.loc[:,'Year'] == (i + 1)
             df.loc[bools,'Prepayment Penalty'] = df.loc[bools,'Balance'] * float(char)/100.
             
    
    # Check to receive when sold - not a function of how much has been spent.
    df.loc[:,'Sale Proceeds'] = df.loc[:,'Home Value'] * (1 - sellCost/100.) - df.loc[:,'Balance'] - df.loc[:,'Prepayment Penalty']
    
    #Maintenance
    df.loc[:,'Maintenance Cost'] = betaprime.rvs(maint_alpha, maint_beta, size=360)*maint_scale * df.loc[:,'Inflation Index']
    
    z3 += (dt.datetime.now() - t1)
    t1 = dt.datetime.now()            
   
    month = 0
    while month < 360:
        df.loc[month,'New Tenant'] = 1
        df.loc[month,'Rent'] = df.loc[month,'Market Rent']        
        tenantStay = np.round((betaprime.rvs(stay_alpha, stay_beta, size=1)*stay_scale)[0]+turnLength,0)
        for tempMonth in range(month,min(month+int(tenantStay),360),12):
            df.loc[tempMonth,'Rent'] = df.loc[tempMonth,'Market Rent']        
    

        month += int(tenantStay)
    
    df.loc[:,'Rent'] = df.loc[:,'Rent'].fillna(method='ffill')
    if pmBool == 'Yes':
        df.loc[df.loc[:,'New Tenant'] == 1,'Turn Costs'] += pmLeaseFee/100. * df.loc[df.loc[:,'New Tenant'] == 1,'Rent']            
                
                
    z4 += (dt.datetime.now() - t1)
    t1 = dt.datetime.now()                
    
    
    df.loc[df.loc[:,'New Tenant']==1,'Vacancy'] = df.loc[df.loc[:,'New Tenant']==1,'Rent'] * turnLength
    
    df.loc[:,'Costs ex loan'] = df.loc[:,'Inflation Index'] * (tax + insurance + otherCosts) / 12. + df.loc[:,['Maintenance Cost','Vacancy']].sum(axis=1)
    if pmBool == 'Yes':
        df.loc[:,'Costs ex loan'] += df.loc[:,'Inflation Index'] * pmFeeFlat / 12. + df.loc[:,'Rent'] * pmFeePerc / 100.
    
    
    #The following 3 metrics are inflation adjusted
    df.loc[:,'Cash Flow'] = (df.loc[:,'Rent'] - df.loc[:,'Mortgage Payment'] - df.loc[:,'Costs ex loan']) / df.loc[:,'Inflation Index']
    df.loc[:,'Cumulative Cash Flow'] = df.loc[:,'Cash Flow'].cumsum()
    df.loc[:,'Profit if sold'] = df.loc[:,'Cumulative Cash Flow'] + df.loc[:,'Sale Proceeds'] / df.loc[:,'Inflation Index'] - initialCosts
    z5 += (dt.datetime.now() - t1)
    t1 = dt.datetime.now()
    return df.loc[:,'Cash Flow'].values, df.loc[:,['Cash Flow','Year']].groupby('Year').sum().values.reshape(30), df.loc[df.loc[:,'Month']%12==0,'Profit if sold'].values, df.loc[df.loc[:,'Month']%12==0,'Cumulative Cash Flow'].values


numberOfIterations = 100


"""
#First pass for 100 iterations: 46
#Second pass for 100: 45
Created a starter DF, 43
Moved some binomial stuff up, 46
Improved the tenant stay logic, 39
Improved Loan calculations 15
Second loan calc improvement, mortgage payment just once, 12
"""
t1 = dt.datetime.now()
output = list(map(singleIteration,range(numberOfIterations)))

today = dt.datetime.now()
start = dt.date(today.year, today.month, 1) + pd.DateOffset(months=1)
end = start + pd.DateOffset(years=30)
monthlyCashFlow = pd.DataFrame(np.array([x[0] for x in output]),columns=pd.date_range(start,end,freq='M'))
annualCashFlow  = pd.DataFrame(np.array([x[1] for x in output]),columns=range(start.year,start.year+30))
profitIfSold    = pd.DataFrame(np.array([x[2] for x in output]),columns=range(start.year,start.year+30))
cumulativeCashflow =pd.DataFrame(np.array([x[3] for x in output]),columns=range(start.year,start.year+30))

t2 = dt.datetime.now()
print((t2-t1).seconds)
annualCashFlow.plot.box()


import matplotlib.pyplot as plt


#Monthly Cash Flow
fig, ax = plt.subplots()
monthlyCashFlow.median().plot(ax=ax)
ax.fill_between(monthlyCashFlow.columns, np.percentile(monthlyCashFlow,10.,axis=0), np.percentile(monthlyCashFlow,90.,axis=0), color="b", alpha=0.2)
ax.fill_between(monthlyCashFlow.columns, np.percentile(monthlyCashFlow,2.5,axis=0), np.percentile(monthlyCashFlow,97.5,axis=0), color="r", alpha=0.1)
ax.set_title('Monthly Cash Flow')
yfmt = '$'+'%.0f'
yticks = mtick.FormatStrFormatter(yfmt)
ax.yaxis.set_major_formatter(yticks)
ax.legend(['Median','80% CI','95% CI'])
fig1 = convertFigureToHTML(fig)



#Total Profit
fig, ax = plt.subplots()
profitIfSold.median().plot(ax=ax)
ax.fill_between(profitIfSold.columns, np.percentile(profitIfSold,10.,axis=0), np.percentile(profitIfSold,90.,axis=0), color="b", alpha=0.2)
ax.fill_between(profitIfSold.columns, np.percentile(profitIfSold,2.5,axis=0), np.percentile(profitIfSold,97.5,axis=0), color="r", alpha=0.1)
ax.set_title('Profit If Sold')
yfmt = '$'+'%.0f'
yticks = mtick.FormatStrFormatter(yfmt)
ax.yaxis.set_major_formatter(yticks)
ax.legend(['Median','80% CI','95% CI'])
fig2 = convertFigureToHTML(fig)

outputTable = np.zeros([30,12],dtype=object)
# Annual Cash Flow
outputTable[:,0] = usd(np.percentile(annualCashFlow, 20, axis=0))
outputTable[:,1] = usd(np.percentile(annualCashFlow, 50, axis=0))
outputTable[:,2] = usd(np.percentile(annualCashFlow, 80, axis=0))

# Profit If Sold
outputTable[:,3] = usd(np.percentile(profitIfSold, 20, axis=0))
outputTable[:,4] = usd(np.percentile(profitIfSold, 50, axis=0))
outputTable[:,5] = usd(np.percentile(profitIfSold, 80, axis=0))

# Cumulative Cash on Cash Return (%) (Annualized)

outputTable[:,6] = percent(np.percentile(cumulativeCashflow, 20, axis=0) / range(1,31) /initialCosts)
outputTable[:,7] = percent(np.percentile(cumulativeCashflow, 50, axis=0) / range(1,31) /initialCosts)
outputTable[:,8] = percent(np.percentile(cumulativeCashflow, 80, axis=0) / range(1,31) /initialCosts)

# Total Return (IRR) If Sold

outputTable[:,9] = percent(np.percentile(profitIfSold, 20, axis=0) / range(1,31) /initialCosts)
outputTable[:,10] = percent(np.percentile(profitIfSold, 50, axis=0) / range(1,31) /initialCosts)
outputTable[:,11] = percent(np.percentile(profitIfSold, 80, axis=0) / range(1,31) /initialCosts)

