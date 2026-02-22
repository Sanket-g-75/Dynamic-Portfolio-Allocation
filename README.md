- The idea is to create a Dynamic Portfolio Allocation that can use data related to indexes, commodities, vix, to adjust portfolio allocation in a set of stocks to maximize the reward 
- Will use YAHOO Finance to download the data of all the scripts.

stocks = ['NVDA','TSLA','RTX','GOOG','XOM','JPM','UNH']
w_indices = ['GSPC','SS','N225','DAX','NSEI','FTSE','FCHI']
forex = ['EURUSD','JPYUSD','GBPUSD','CHFUSD','AUDUSD']
commodities = ['GC','SI','CL','HG']
other = ['VIX']