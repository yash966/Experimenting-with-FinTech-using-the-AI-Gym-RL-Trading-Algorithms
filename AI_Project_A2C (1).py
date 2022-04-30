#!/usr/bin/env python
# coding: utf-8

# ## Installation of libraries of specific version

# In[1]:


pip install tensorflow==1.15


# In[2]:


pip install stable_baselines


# In[4]:


pip install mpl_finance


# In[3]:


pip install matplotlib==3.1


# ## Import Libraries 

# In[25]:



import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import pandas as pd
import datetime as dt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


# ## Required packaged to run code

# In[26]:


from stable_baselines import A2C
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style
from mpl_finance import candlestick_ochl as candlestick


# ## Read Data from CSV

# In[27]:



df = pd.read_csv('MSFT.csv')
df = df.sort_values('Date')


# ## parameters for graph

# In[ ]:


style.use('seaborn-darkgrid')

VOLUME_CHART_HEIGHT = 0.33

UP_COLOR = '#5CFF33'
DOWN_COLOR = '#EF534F'
UP_TEXT_COLOR = '#339EFF'
DOWN_TEXT_COLOR = '#DC2C27'
def date2num(date):
    converter = mdates.strpdate2num('%Y-%m-%d')
    return converter(date)


# ## graph trading 

# In[28]:



class StockTradingGraph:
    """A stock trading visualization using matplotlib made to render OpenAI gym environments"""

    def __init__(self, df, title="Microsoft"):
        self.df = df
        self.nws = np.zeros(len(df['Date']))   

        
        fig = plt.figure()
        fig.suptitle(title)

        
        self.nw_ax = plt.subplot2grid(
            (6, 1), (0, 0), rowspan=2, colspan=1)

        
        self.price_ax = plt.subplot2grid(
            (6, 1), (2, 0), rowspan=8, colspan=1, sharex=self.nw_ax)

        
        self.volume_ax = self.price_ax.twinx()

        # Add padding to make graph easier to view
        plt.subplots_adjust(left=0.11, bottom=0.24,
                            right=0.90, top=0.90, wspace=0.2, hspace=0)

        
        plt.ion()

    def render_income(self, cs, nw, sr, dates):        
        # Clear the frame rendered last step
        self.nw_ax.clear()

        # Plot net worths
        self.nw_ax.plot_date(
            dates, self.nws[sr], '-', label='Net Worth')

        
        self.nw_ax.legend()
        legend = self.nw_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        last_date = date2num(self.df['Date'].values[cs])
        last_nw = self.nws[cs]

       
        self.nw_ax.annotate('{0:.2f}'.format(nw), (last_date, last_nw),
                                   xytext=(last_date, last_nw),
                                   bbox=dict(boxstyle='round',
                                             fc='w', ec='k', lw=1),
                                   color="black",
                                   fontsize="small")

        # Add space above and below min/max net worth
        self.nw_ax.set_ylim(
            min(self.nws[np.nonzero(self.nws)]) / 1.25, max(self.nws) * 1.25)

    def ren_prc(self, cs, nw, dates, sr):        
        self.price_ax.clear()

       
        candlesticks = zip(dates,
                           self.df['Open'].values[sr], self.df['Close'].values[sr],
                           self.df['High'].values[sr], self.df['Low'].values[sr])

        # Plot price using candlestick graph from mpl_finance
        candlestick(self.price_ax, candlesticks, width=1,
                    colorup=UP_COLOR, colordown=DOWN_COLOR)

        last_date = date2num(self.df['Date'].values[cs])
        last_close = self.df['Close'].values[cs]
        last_high = self.df['High'].values[cs]

        
        self.price_ax.annotate('{0:.2f}'.format(last_close), (last_date, last_close),
                               xytext=(last_date, last_high),
                               bbox=dict(boxstyle='round',
                                         fc='w', ec='k', lw=1),
                               color="black",
                               fontsize="small")

        # Shift price axis up to give volume chart space
        ylim = self.price_ax.get_ylim()
        self.price_ax.set_ylim(ylim[0] - (ylim[1] - ylim[0])
                               * VOLUME_CHART_HEIGHT, ylim[1])

    def ren_vlm(self, cs, nw, dates, sr):        
        self.volume_ax.clear()

        volume = np.array(self.df['Volume'].values[sr])

        pos = self.df['Open'].values[sr] -             self.df['Close'].values[sr] < 0
        neg = self.df['Open'].values[sr] -             self.df['Close'].values[sr] > 0

        # Color volume bars based on price direction on that date
        self.volume_ax.bar(dates[pos], volume[pos], color=UP_COLOR,
                           alpha=0.4, width=1, align='center')
        self.volume_ax.bar(dates[neg], volume[neg], color=DOWN_COLOR,
                           alpha=0.4, width=1, align='center')

        
        self.volume_ax.set_ylim(0, max(volume) / VOLUME_CHART_HEIGHT)
        self.volume_ax.yaxis.set_ticks([])

    def ren_trd(self, cs, trades, sr):        
        for trade in trades:
            if trade['step'] in sr:
                date = date2num(self.df['Date'].values[trade['step']])
                high = self.df['High'].values[trade['step']]
                low = self.df['Low'].values[trade['step']]

                if trade['type'] == 'buy':
                    high_low = low
                    color = UP_TEXT_COLOR
                else:
                    high_low = high
                    color = DOWN_TEXT_COLOR

                total = '{0:.2f}'.format(trade['total'])

                # Print the current price to the price axis
                self.price_ax.annotate(f'${total}', (date, high_low),
                                       xytext=(date, high_low),
                                       color=color,
                                       fontsize=8,
                                       arrowprops=(dict(color=color)))

    def render(self, cs, nw, trades, window_size=40):        
        self.nws[cs] = nw        
        window_start = max(cs - window_size, 0)
        sr = range(window_start, cs + 1)        
        # Format dates as timestamps, necessary for candlestick graph
        dates = np.array([date2num(x)
                          for x in self.df['Date'].values[sr]])

        self.render_income(cs, nw, sr, dates)
        self.ren_prc(cs, nw, dates, sr)
        self.ren_vlm(cs, nw, dates, sr)
        self.ren_trd(cs, trades, sr)

       
        self.price_ax.set_xticklabels(self.df['Date'].values[sr], rotation=45,
                                      horizontalalignment='right')

        # Hide duplicate net worth date labels
        plt.setp(self.nw_ax.get_xticklabels(), visible=False)

        # Necessary to view frames before they are unrendered
        plt.pause(0.001)

    def close(self):
        plt.close()


# ## Initial Parameters for Environment

# In[29]:



MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 100000

LOOKBACK_WINDOW_SIZE = 15


# In[32]:


def funcfp(val):
    return [(i, val / i) for i in range(1, int(val**0.5)+1) if val % i == 0]


class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['live', 'file', 'none']}
    visualization = None

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = self.funcflex_price(df)
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy, Sell, Hold, the shares.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the Open, High, Close, Low values for given past days
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, LOOKBACK_WINDOW_SIZE + 2), dtype=np.float16)

    def funcflex_price(self, df):
        adjust_ratio = df['Adj Close'] / df['Close']

        df['Open'] = df['Open'] * adjust_ratio
        df['High'] = df['High'] * adjust_ratio
        df['Low'] = df['Low'] * adjust_ratio
        df['Close'] = df['Close'] * adjust_ratio

        return df

    def func_for_obs(self):
        frame = np.zeros((5, LOOKBACK_WINDOW_SIZE + 1))

        
        np.put(frame, [0, 4], [
            self.df.loc[self.cs: self.cs +
                        LOOKBACK_WINDOW_SIZE, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.cs: self.cs +
                        LOOKBACK_WINDOW_SIZE, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.cs: self.cs +
                        LOOKBACK_WINDOW_SIZE, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.cs: self.cs +
                        LOOKBACK_WINDOW_SIZE, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.cs: self.cs +
                        LOOKBACK_WINDOW_SIZE, 'Volume'].values / MAX_NUM_SHARES,
        ])

        # Append additional data and scale each value to between 0-1 to make observation
        obs = np.append(frame, [
            [self.balance / MAX_ACCOUNT_BALANCE],
            [self.max_nw / MAX_ACCOUNT_BALANCE],
            [self.shares_held / MAX_NUM_SHARES],
            [self.cost_basis / MAX_SHARE_PRICE],
            [self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE)],
        ], axis=1)

        return obs
    
   
    def _take_action(self, action):
        current_price = random.uniform(
            self.df.loc[self.cs, "Open"], self.df.loc[self.cs, "Close"])

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

            if shares_bought > 0:
                self.trades.append({'step': self.cs,
                                    'shares': shares_bought, 'total': additional_cost,
                                    'type': "buy"})

        elif action_type < 2:
           
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

            if shares_sold > 0:
                self.trades.append({'step': self.cs,
                                    'shares': shares_sold, 'total': shares_sold * current_price,
                                    'type': "sell"})

        self.nw = self.balance + self.shares_held * current_price

        if self.nw > self.max_nw:
            self.max_nw = self.nw

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.cs += 1

        delay_modifier = (self.cs / MAX_STEPS)

        reward = self.balance * delay_modifier + self.cs
        done = self.nw <= 0 or self.cs >= len(
            self.df.loc[:, 'Open'].values)

        obs = self.func_for_obs()

        return obs, reward, done, {}
    
    #Reset the environment to initial state
    def reset(self):
       
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.nw = INITIAL_ACCOUNT_BALANCE
        self.max_nw = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.cs = 0
        self.trades = []

        return self.func_for_obs()

    #Used to render data to display
    def render(self, mode='live', **kwargs):
        profit = self.nw - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.cs}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.nw} (Max net worth: {self.max_nw})')
        print(f'Profit: {profit}')
        
        #Call visualization class to create visualization
        if self.visualization == None:
            self.visualization = StockTradingGraph(self.df)

        if self.cs > LOOKBACK_WINDOW_SIZE:
            self.visualization.render(self.cs, self.nw, self.trades, window_size=LOOKBACK_WINDOW_SIZE)

    def close(self):
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None


# ## The algorithms require a vectorized environment to run

# In[33]:


# 
env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000)

obs = env.reset()
for i in range(200):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()


# In[ ]:





# In[ ]:




