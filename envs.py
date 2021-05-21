import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools


class TradingEnv(gym.Env):
    """
    A 3-stock (MSFT, IBM, QCOM) trading environment.

    State: [# of stock owned, current stock prices, cash in hand]
      - array of length n_stock * 2 + 1
      - price is discretized (to integer) to reduce state space
      - use close price for each stock
      - cash in hand is evaluated at each step based on action performed

    Action: sell (0), hold (1), and buy (2)
      - when selling, sell all the shares
      - when buying, buy as many as cash in hand allows
      - if buying multiple stock, equally distribute cash in hand and then utilize the balance
    """

    def __init__(self, train_data, init_invest=20000):
        # data
        self.stock_price_history = np.around(train_data)  # round up to integer to reduce state space
        self.n_stock, self.n_step, _ = self.stock_price_history.shape

        # instance attributes
        self.init_invest = init_invest
        self.cur_step = None
        self.stock_owned = None

        # features
        self.stock_price = None
        self.stock_rmo = None
        self.stock_r2mo = None
        self.stock_r3mo = None
        self.stock_ryear = None
        self.stock_macd = None
        self.stock_rsi = None

        self.cash_in_hand = None

        # action space
        self.action_space = spaces.Discrete(3 ** self.n_stock)

        # observation space: give estimates in order to sample and build scaler
        stock_max_price = self.stock_price_history.max(axis=1)[:, 0]
        stock_max_rmo = self.stock_price_history.max(axis=1)[:, 1]
        stock_max_r2mo = self.stock_price_history.max(axis=1)[:, 2]
        stock_max_r3mo = self.stock_price_history.max(axis=1)[:, 3]
        stock_max_ryear = self.stock_price_history.max(axis=1)[:, 4]
        stock_max_macd = self.stock_price_history.max(axis=1)[:, 5]
        stock_max_rsi = self.stock_price_history.max(axis=1)[:, 6]

        stock_range = [[0, init_invest * 2 // mx] for mx in stock_max_price]

        price_range = [[0, mx] for mx in stock_max_price]
        rmo_range = [[0, mx] for mx in stock_max_rmo]
        r2mo_range = [[0, mx] for mx in stock_max_r2mo]
        r3mo_range = [[0, mx] for mx in stock_max_r3mo]
        ryear_range = [[0, mx] for mx in stock_max_ryear]
        macd_range = [[0, mx] for mx in stock_max_macd]
        rsi_range = [[0, mx] for mx in stock_max_rsi]

        cash_in_hand_range = [[0, init_invest * 2]]

        self.observation_space = spaces.MultiDiscrete(stock_range + price_range + rmo_range + r2mo_range + r3mo_range +
                                                      ryear_range + macd_range + rsi_range + cash_in_hand_range)

        self.portfolio_history=[]
        # seed and start
        self._seed()
        self._reset()

        #previous action


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.cur_step = 0
        self.stock_owned = [0] * self.n_stock

        self.stock_price = self.stock_price_history[:, self.cur_step, 0]

        self.stock_rmo = self.stock_price_history[:, self.cur_step, 1]
        self.stock_r2mo = self.stock_price_history[:, self.cur_step, 2]
        self.stock_r3mo = self.stock_price_history[:, self.cur_step, 3]
        self.stock_ryear = self.stock_price_history[:, self.cur_step, 4]
        self.stock_macd = self.stock_price_history[:, self.cur_step, 5]
        self.stock_rsi = self.stock_price_history[:, self.cur_step, 6]

        self.cash_in_hand = self.init_invest
        return self._get_obs()

    def _step(self, action):
        assert self.action_space.contains(action)
        prev_val = self._get_val()
        self.cur_step += 1

        # update features
        self.stock_price = self.stock_price_history[:, self.cur_step, 0]
        self.stock_rmo = self.stock_price_history[:, self.cur_step, 1]
        self.stock_r2mo = self.stock_price_history[:, self.cur_step, 2]
        self.stock_r3mo = self.stock_price_history[:, self.cur_step, 3]
        self.stock_ryear = self.stock_price_history[:, self.cur_step, 4]
        self.stock_macd = self.stock_price_history[:, self.cur_step, 5]
        self.stock_rsi = self.stock_price_history[:, self.cur_step, 6]

        self._trade(action)

        cur_val = self._get_val()
        reward = cur_val - prev_val
        done = self.cur_step == self.n_step - 1
        info = {'cur_val': cur_val}

        self.portfolio_history.append(cur_val)

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        obs = []
        obs.extend(self.stock_owned)
        obs.extend(list(self.stock_price))
        obs.extend(list(self.stock_rmo))
        obs.extend(list(self.stock_r2mo))
        obs.extend(list(self.stock_r3mo))
        obs.extend(list(self.stock_ryear))
        obs.extend(list(self.stock_macd))
        obs.extend(list(self.stock_rsi))

        obs.append(self.cash_in_hand)
        return obs

    def _get_val(self):
        return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand

    def _trade(self, action):
        # all combo to sell(0), hold(1), or buy(2) stocks
        action_combo = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
        action_vec = action_combo[action]

        # one pass to get sell/buy index
        sell_index = []
        buy_index = []
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        # two passes: sell first, then buy; might be naive in real-world settings
        if sell_index:
            for i in sell_index:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0
        if buy_index:
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1  # buy one share
                        self.cash_in_hand -= self.stock_price[i]
                    else:
                        can_buy = False
