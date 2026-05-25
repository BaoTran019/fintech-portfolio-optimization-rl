import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv

class StockPortfolioEnv(gym.Env):
    """
    Gym-compatible environment for stock portfolio allocation using RL.
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                df,
                stock_dim,
                hmax,
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold=None,
                lookback=252,
                day=0,
                seed=None):
        super().__init__()
        self.day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.excess_return_memory = [0]

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,)) 
        # covariance matrix + technical indicators
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space + len(self.tech_indicator_list), self.state_space))

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.covs = self.data['cov_list'].values[0]
        self.state = np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list], axis=0)
        self.terminal = False     
        self.turbulence_threshold = turbulence_threshold        
        # initialize state: initial portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1/self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]

        # Set seed
        if seed is not None:
            self.seed(seed)
        
    def step(self, actions):
        # Check if terminal
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']
            plt.plot(df.daily_return.cumsum(), 'r')
            plt.savefig('results/cumulative_reward.png')
            plt.close()
            
            plt.plot(self.portfolio_return_memory, 'r')
            plt.savefig('results/rewards.png')
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))           
            print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ['daily_return']
            if df_daily_return['daily_return'].std() != 0:
                sharpe = (252**0.5) * df_daily_return['daily_return'].mean() / df_daily_return['daily_return'].std()
                print("Sharpe: ", sharpe)
            print("=================================")
            
            return self.state, self.reward, self.terminal, {}

        else:
            # Normalize actions to sum to 1
            weights = self.softmax_normalization(actions)

            # Calculate portfolio turnover

            previous_weights = self.actions_memory[-1]

            turnover = np.sum(
                np.abs(weights - previous_weights)
            )

            transaction_cost = (
                turnover * self.transaction_cost_pct
            ) 
            self.actions_memory.append(weights)

            last_day_memory = self.data

            # Load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            
            if isinstance(self.data, pd.Series):
                self.covs = self.data['cov_list']
            else:
                self.covs = self.data['cov_list'].values[0]
                
            # Handle indicators flexibly
            tech_data_list = []
            for tech in self.tech_indicator_list:
                if isinstance(self.data, pd.Series):
                    tech_data_list.append([self.data[tech]])  # Wrap single value in list
                else:
                    tech_data_list.append(self.data[tech].values.tolist())
                
            self.state = np.append(np.array(self.covs), tech_data_list, axis=0)
            
            # Calculate portfolio return
            portfolio_return = np.sum(
                ((self.data.close.values /
                last_day_memory.close.values) - 1)
                * weights
            )

            portfolio_return = (
                portfolio_return - transaction_cost
            )

            if isinstance(self.data, pd.Series):
                benchmark_return = self.data['vn30_return']
            else:
                benchmark_return = self.data['vn30_return'].values[0]

            excess_return = portfolio_return - benchmark_return
            self.excess_return_memory.append(excess_return)

            # Update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value

            # Save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)

            # Reward calculation
            lookback_window = 63

            if len(self.excess_return_memory) >= lookback_window:

                recent_excess_returns = np.array(
                    self.excess_return_memory[-lookback_window:]
                )

                volatility = (
                    recent_excess_returns.std() + 1e-6
                )

                reward = excess_return / volatility

            else:

                reward = excess_return

            self.reward = reward * self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.excess_return_memory = [0]
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        # Load states
        if isinstance(self.data, pd.Series):
            self.covs = self.data['cov_list']
        else:
            self.covs = self.data['cov_list'].values[0]
            
        # Handle indicators flexibly
        tech_data_list = []
        for tech in self.tech_indicator_list:
            if isinstance(self.data, pd.Series):
                tech_data_list.append([self.data[tech]])  # Wrap single value in list
            else:
                tech_data_list.append(self.data[tech].values.tolist())
                
        self.state = np.append(np.array(self.covs), tech_data_list, axis=0)
        self.portfolio_value = self.initial_amount
        self.terminal = False 
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1/self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]] 
        return self.state
    
    def render(self, mode='human'):
        return self.state
        
    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    
    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame({'date': date_list, 'daily_return': portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # Date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']
        
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        return df_actions

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs