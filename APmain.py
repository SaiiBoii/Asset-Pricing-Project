import pandas as pd
import yfinance as yf
import numpy as np
import math as m
import matplotlib.pyplot as plt
import random as r


risk_free_rate=0.03
tickers=['DHR','GOOGL','PM','^GSPC'] #tickers assigned

stocks_df=yf.download(tickers)['Close'] #filtering assigned tickers from the excel file
stocks_df.dropna(inplace=True) #dropping any missing values
monthly_simple_returns=stocks_df.pct_change()    
annualized_mean_return=monthly_simple_returns.mean()*365 #annualizing mean pct. changes
annualized_standard_deviation=monthly_simple_returns.std()*np.sqrt(365) #annual standdard deviation

#calculating excess monthly returns
excess_returns_df=(monthly_simple_returns)-(risk_free_rate/365)

#calculating variables to calculate the sharpe ratio
mean_annual_excess_return=excess_returns_df.mean()*365
mean_annual_excess_standard_deviation=excess_returns_df.std()*np.sqrt(365)
sharpe_ratio=mean_annual_excess_return/mean_annual_excess_standard_deviation

#maximum and minimum prices
maximum_price=stocks_df.max()
minimum_price=stocks_df.min()


#consolidating all part 1 information into a dataframe
finaltable_df=pd.DataFrame({'Average annual return':annualized_mean_return,
                            'Annual standard deviation':annualized_standard_deviation,
                            'Mean annual excess return':mean_annual_excess_return,
                            'Sharpe_ratio':sharpe_ratio,
                            'Maximum prices':maximum_price,
                            'Minimum prices':minimum_price})

#annualized covariance matrix for parts 2,3,4
covariance_matrix=monthly_simple_returns.cov()*365
correlation_matrix=monthly_simple_returns.corr() #correlation matrix    



#plotting monthly pct. change
plt.figure(figsize=(16,10))
plt.title(label='Monthly Returns Plot')
for stock in stocks_df.columns:
    plt.plot(monthly_simple_returns[stock],lw=1.5)
plt.xlabel('Date')
plt.ylabel('Monthly Returns')
plt.legend(stocks_df.columns,loc='upper left')
plt.grid()
plt.show()


final_stocks_df=(stocks_df)/(stocks_df.iloc[0])*100 #investing 100 dollars in each stock


#plotting price chart
plt.figure(figsize=(16,9))
plt.title(label='Price Chart')
for stock in final_stocks_df.columns:
    plt.plot(final_stocks_df[stock],lw=4)   

plt.xlabel('Date')
plt.ylabel('Value of $100 Invested')
plt.grid()
plt.legend(final_stocks_df.columns, loc='upper left')



#plotting covid-19 price drop
covid_crash_df=yf.download(tickers,start='2020-02-01',end='2020-04-01')['Close']

drawdown_df=pd.DataFrame({'High Price': covid_crash_df.max(),
                          'Low Price': covid_crash_df.min()},index=tickers)
drawdown_df['Drawdown']=(drawdown_df['High Price']-drawdown_df['Low Price'])/drawdown_df['High Price']

covid_crash_df=(covid_crash_df/covid_crash_df.iloc[0])*100

plt.figure(figsize=(16,9))
plt.title(label='Covid Crash Plot')
for stock in covid_crash_df.columns:
    plt.plot(covid_crash_df[stock],lw=4)

plt.xlabel('Date')
plt.ylabel('Value of $100 invested')
plt.legend(covid_crash_df.columns,loc='upper left')
plt.grid()
plt.show()

#Boxplot
monthly_simple_returns.boxplot(xlabel='Tickers',ylabel='Monthly percent change',figsize=(7,7))
plt.title(label='Monthly returns boxplot')
plt.locator_params(axis='y',nbins=35)
plt.show()

#printing results
print(covariance_matrix)
print(correlation_matrix)
print(finaltable_df)
print(drawdown_df)


stocks=['DHR','GOOGL','PM']

covariance_matrix=monthly_simple_returns[stocks].cov()*365
annual_mean_return=annualized_mean_return[stocks]
excess_returns=annual_mean_return-risk_free_rate
inv_covariance_matrix=np.linalg.inv(covariance_matrix)
ones=np.ones(3)
min_var_weights = np.dot(inv_covariance_matrix, ones)/np.dot(ones.T, np.dot(inv_covariance_matrix, ones))
optimal_risky_weights=np.dot(inv_covariance_matrix,excess_returns)/np.dot(ones.T,np.dot(inv_covariance_matrix,excess_returns))


def portfolio_metrics(w):

    tickers=['DHR','GOOGL','PM'] #tickers list without the index
    mean_returns=annualized_mean_return[tickers] #dataframe calculated in part 1, filtering out the index
    covariance_matrix=monthly_simple_returns[tickers].cov()*354 #cov matrix without index
    portfolio_return=np.dot(np.array(w),mean_returns) 
    portfolio_variance = np.dot(np.array(w).T, np.dot(covariance_matrix, np.array(w))) #covariance matrix calculated in part 1
    portfolio_std = np.sqrt(portfolio_variance)

    sharpe_ratio=(portfolio_return-risk_free_rate)/portfolio_std

    metrics_dict={'Tickers':tickers,
                  'Weights':w,
                  'Return':portfolio_return,
                  'Risk':portfolio_std,
                  'Sharpe Ratio':sharpe_ratio}
    
    return metrics_dict




weights_list = []
portfolio_returns = []
portfolio_risks = []
sharpe_ratio_list = []

for w1 in np.linspace(0, 1, 100):
    for w2 in np.linspace(0, 1 - w1, 100):
        w3=1-w1-w2  
        weight = [w1, w2, w3]

        weights_list.append(weight)
        portfolio_returns.append(portfolio_metrics(weight)['Return'])
        portfolio_risks.append(portfolio_metrics(weight)['Risk'])
        sharpe_ratio_list.append(portfolio_metrics(weight)['Sharpe Ratio'])

s1_risk=portfolio_metrics(np.array([1,0,0]))['Risk']
s1_return=portfolio_metrics(np.array([1,0,0]))['Return']

s2_risk=portfolio_metrics(np.array([0,1,0]))['Risk']
s2_return=portfolio_metrics(np.array([0,1,0]))['Return']

s3_risk=portfolio_metrics(np.array([0,0,1]))['Risk']
s3_return=portfolio_metrics(np.array([0,0,1]))['Return']

plt.figure(figsize=(16, 9))
plt.gca().set_facecolor('#f5f5f5')
plt.scatter(portfolio_risks, portfolio_returns, c=sharpe_ratio_list,cmap='gnuplot' ,marker='o')
plt.colorbar(label='Sharpe Ratio')
plt.locator_params(axis='x',nbins=25)
plt.locator_params(axis='y',nbins=25)
plt.xticks(rotation=90)
plt.xlabel("Portfolio Standard Deviation (Risk)")
plt.ylabel("Portfolio Expected Return")
plt.title("Investor's Opportunity Set")
plt.scatter(portfolio_metrics(min_var_weights)['Risk'],portfolio_metrics(min_var_weights)['Return'],color='Blue',label='Global Minimum',marker='*',s=200)
plt.scatter(portfolio_metrics(optimal_risky_weights)['Risk'],portfolio_metrics(optimal_risky_weights)['Return'],color='Black',label='Global Optimal',marker='*',s=200)
plt.scatter(s1_risk,s1_return,color='Red',label=stocks[0],marker='*',s=200)
plt.scatter(s2_risk,s2_return,color='Orange',label=stocks[1],marker='*',s=200)
plt.scatter(s3_risk,s3_return,color='Green',label=stocks[2],marker='*',s=200)
plt.legend()
plt.grid()
plt.show()

min_var_portfolio=portfolio_metrics(min_var_weights)
optimal_risky_portfolio=portfolio_metrics(optimal_risky_weights)

print(min_var_portfolio)
print(optimal_risky_portfolio)




random_weights_list=[]
random_returns_list=[]
random_risk_list=[]
random_sharpe_list=[]


for _ in range(0,10):

    random_weight=r.choice(weights_list)

    random_weight=np.array([round(weight,3) for weight in list(random_weight)])

    random_metrics=portfolio_metrics(random_weight)

    random_weights_list.append(random_weight)
    random_returns_list.append(random_metrics['Return'])
    random_risk_list.append(random_metrics['Risk'])
    random_sharpe_list.append(random_metrics['Sharpe Ratio'])


random_metrics_df=pd.DataFrame({'Weights':random_weights_list,
                                'Returns':random_returns_list,
                                'Risk':random_risk_list,
                                'Sharpe Ratio':random_sharpe_list})

print(random_metrics_df)

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.api import add_constant, OLS


tickers=['DHR','GOOGL','PM']
alpha_dict={}
beta_dict={}
residual_variance_dict={}
residual_mean_dict={}
residuals_dict={}

for ticker in tickers:

    y = excess_returns_df[ticker].dropna()
    x = excess_returns_df['^GSPC'].dropna()

    x_constant = add_constant(x)


    model = OLS(y, x_constant).fit()

    print(model.summary())

    residual_variance=model.mse_resid
    residual_mean=model.resid.mean()
    alpha,beta = model.params
    residuals_dict[ticker]=model.resid
    residual_mean_dict[ticker]=residual_mean
    alpha_dict[ticker]=alpha
    beta_dict[ticker]=beta
    residual_variance_dict[ticker]=residual_variance




    plt.figure(figsize=(16, 9))
    plt.scatter(x, y, color='black')


    regression_line = alpha + beta*np.linspace(min(x), max(x), 1000)
    plt.plot(np.linspace(min(x), max(x), 1000), regression_line, color='blue')
    plt.locator_params(axis='x',nbins=50)
    plt.locator_params(axis='y',nbins=50)

    plt.xlabel("^GSPC excess returns")
    plt.ylabel(f"{ticker} excess returns")
    plt.title("Single Index Model")
    plt.grid(True)
    plt.show()

market_variance=x.var()
stocks=['DHR','GOOGL','PM']
dhr_beta=beta_dict[stocks[0]]
googl_beta=beta_dict[stocks[1]]
pm_beta=beta_dict[stocks[2]]

dhr_residuals_var=residual_variance_dict[stocks[0]]
googl_residuals_var=residual_variance_dict[stocks[1]]
pm_residuals_var=residual_variance_dict[stocks[2]]

cov_11=dhr_beta*dhr_beta*market_variance+dhr_residuals_var
cov_12=dhr_beta*googl_beta*market_variance
cov_13=dhr_beta*pm_beta*market_variance

cov_21=cov_12
cov_22=googl_beta*googl_beta*market_variance+googl_residuals_var
cov_23=googl_beta*pm_beta*market_variance

cov_31=cov_13
cov_32=cov_23
cov_33=pm_beta*pm_beta*market_variance+pm_residuals_var


sim_cov_matrix = np.array([
    [cov_11, cov_12, cov_13],
    [cov_21, cov_22, cov_23],
    [cov_31, cov_32, cov_33]
])

sim_cov_matrix

print(sim_cov_matrix*12)




def sim_expected_return(ticker):

    alpha=alpha_dict[ticker]
    beta=beta_dict[ticker]
    residual=residual_mean_dict[ticker]
    r_f=0.03/12
    market_excess_return=x.mean()
    expected_excess_return=alpha + beta*market_excess_return + residual
    expected_return=expected_excess_return

    return expected_return*12



sim_expected_return('PM'),annualized_standard_deviation**2


stocks=['DHR','GOOGL','PM']
annual_mean_return=np.array([sim_expected_return(stock) for stock in stocks])


excess_returns=annual_mean_return
inv_covariance_matrix=np.linalg.inv(sim_cov_matrix)
ones=np.ones(3)
sim_min_var_weights = np.dot(inv_covariance_matrix, ones)/np.dot(ones.T, np.dot(inv_covariance_matrix, ones))
sim_optimal_risky_weights=np.dot(inv_covariance_matrix,excess_returns)/np.dot(ones.T,np.dot(inv_covariance_matrix,excess_returns))





def sim_portfolio_metrics(w):

    tickers=['DHR','GOOGL','PM']
    portfolio_return=0
    weight_dict=dict(zip(tickers,w))

    for ticker in tickers:
        wi,ai,bi,rm,ei=weight_dict[ticker],alpha_dict[ticker],beta_dict[ticker],(x).mean(),residual_mean_dict[ticker]
        portfolio_return+=((wi*(ai+bi*rm+ei)))
    
    portfolio_return*=12

    beta=0
    market_variance=x.var()
    residual_variance=0
    for ticker in tickers:
        beta+=(weight_dict[ticker]*beta_dict[ticker])
        residual_variance+=(weight_dict[ticker]**2)*(residual_variance_dict[ticker]*12)
        
    beta2=beta**2

    portfolio_variance=beta2*market_variance*12+residual_variance

    portfolio_std=np.sqrt(portfolio_variance)

    sharpe_ratio=(portfolio_return)/portfolio_std

    metrics_dict={'Tickers':tickers,
                  'Weights':w,
                  'Return':portfolio_return,
                  'Risk':portfolio_std,
                  'Sharpe Ratio':sharpe_ratio}
    
    return metrics_dict

weights_list = []
portfolio_returns = []
portfolio_risks = []
sharpe_ratio_list = []

for w1 in np.linspace(0, 1, 50):
    for w2 in np.linspace(0, 1 - w1, 50):
        w3=1-w1-w2
        weight = np.array([w1, w2, w3])

        weights_list.append(weight)
        portfolio_returns.append(sim_portfolio_metrics(weight)['Return'])
        portfolio_risks.append(sim_portfolio_metrics(weight)['Risk'])
        sharpe_ratio_list.append(sim_portfolio_metrics(weight)['Sharpe Ratio'])

s1_risk=sim_portfolio_metrics(np.array([1,0,0]))['Risk']
s1_return=sim_portfolio_metrics(np.array([1,0,0]))['Return']

s2_risk=sim_portfolio_metrics(np.array([0,1,0]))['Risk']
s2_return=sim_portfolio_metrics(np.array([0,1,0]))['Return']

s3_risk=sim_portfolio_metrics(np.array([0,0,1]))['Risk']
s3_return=sim_portfolio_metrics(np.array([0,0,1]))['Return']

cmap_list=['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']

plt.figure(figsize=(16, 9))
plt.gca().set_facecolor('#f5f5f5')
plt.scatter(portfolio_risks, portfolio_returns, c=sharpe_ratio_list, cmap='gnuplot',marker='o')
plt.colorbar(label='Sharpe Ratio')
plt.locator_params(axis='x',nbins=25)
plt.locator_params(axis='y',nbins=25)
plt.xticks(rotation=90)
plt.xlabel("Portfolio Standard Deviation (Risk)")
plt.ylabel("Portfolio Expected Excess Return")
plt.title("Investor's Opportunity Set as per Single Index Model")
plt.scatter(sim_portfolio_metrics(sim_min_var_weights)['Risk'],sim_portfolio_metrics(sim_min_var_weights)['Return'],color='Blue',label='Global Minimum',marker='*',s=200)
plt.scatter(sim_portfolio_metrics(sim_optimal_risky_weights)['Risk'],sim_portfolio_metrics(sim_optimal_risky_weights)['Return'],color='Black',label='Global Optimal',marker='*',s=200)
plt.scatter(s1_risk,s1_return,color='Red',label=stocks[0],marker='*',s=200)
plt.scatter(s2_risk,s2_return,color='Orange',label=stocks[1],marker='*',s=200)
plt.scatter(s3_risk,s3_return,color='Green',label=stocks[2],marker='*',s=200)
plt.grid(True)
plt.legend()
plt.show()

sim_min_var_portfolio=portfolio_metrics(sim_min_var_weights)
sim_optimal_risky_portfolio=portfolio_metrics(sim_optimal_risky_weights)

print(sim_min_var_portfolio)
print(sim_optimal_risky_portfolio)







