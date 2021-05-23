import pickle
import time
import numpy as np
import argparse
import re
from envs import TradingEnv
from agent_torch import DQNAgent
from agent_torch_pg import PGAgent
from utils import get_data, get_scaler, maybe_make_dir
from data_handler import *
import datetime as dt


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--episode', type=int, default=1,
                      help='number of episode to run')
  parser.add_argument('-b', '--batch_size', type=int, default=32,
                      help='batch size for experience replay')
  parser.add_argument('-i', '--initial_invest', type=int, default=20000,
                      help='initial investment amount')
  parser.add_argument('-m', '--mode', type=str,  default='train',
                      help='either "train" or "test"')
  parser.add_argument('-md', '--model', type=str, default='DQN',
                      help='model to use')
  parser.add_argument('-lr', '--lr', type=float, default=0.001,
                      help='learning rate')
  parser.add_argument('-w', '--weights', type=str, help='a trained model weights')

  args = parser.parse_args()

  maybe_make_dir('weights')
  maybe_make_dir('portfolio_val')
  maybe_make_dir('portfolio_hist')
  maybe_make_dir('stocks_hist')

  #timestamp = time.strftime('%Y%m%d%H%M')
  timestamp = args.initial_invest

  #start = dt.date(2015, 1, 1)
  #end = dt.date(2021, 1, 1)
  #st = MultiStock(['AAPL', 'NVDA', 'TSLA'])
  #feat, date = st.get_all_features(start, end)
  #train_data=np.around(feat)[:,:-400]
  #test_data = np.around(feat)[:, -400:]

  start = dt.date(2016, 1, 1)
  end = dt.date(2021, 2, 1)
  st = MultiStock(['BCH-USD', 'MKR-USD', 'ETH-USD'])
  feat, date = st.get_all_features(start, end)
  train_data = np.around(feat)[:, :-150]
  test_data = np.around(feat)[:, -150:]

  env = TradingEnv(train_data, args.initial_invest)
  state_size = env.observation_space.shape
  action_size = env.action_space.n
  if args.model=='PG':
    agent = PGAgent(state_size, action_size, model=args.model, lr=args.lr)
  else:
    agent = DQNAgent(state_size, action_size, model=args.model, lr=args.lr)
  scaler = get_scaler(env)

  portfolio_value = []

  if args.mode == 'test':
    env = TradingEnv(test_data, args.initial_invest)
    agent.load(args.weights)
    timestamp = timestamp

  for e in range(args.episode):
    state = env.reset()
    state = scaler.transform([state])
    for time in range(env.n_step):
      action = agent.act(state)
      next_state, reward, done, info = env.step(action)
      a=env.stocks_l
      next_state = scaler.transform([next_state])
      if args.mode == 'train':
        agent.remember(state, action, reward, next_state, done)
      state = next_state
      if done:
        print("episode: {}/{}, episode end value: {}".format(
          e + 1, args.episode, info['cur_val']))
        portfolio_value.append(info['cur_val']) # append episode end portfolio value
        break
      if args.mode == 'train' and len(agent.memory) > args.batch_size:
        agent.replay(args.batch_size)
    if args.mode == 'train' and (e+1) % 50 == 0:  # checkpoint weights
      agent.save('weights/{}-{}.h5'.format(timestamp, args.model))
    with open('stocks_hist/{}-{}-{}.p'.format(timestamp, args.mode, args.model), 'wb') as fp:
      pickle.dump(env.stocks_l, fp)

  # save portfolio value history to disk
  with open('portfolio_val/{}-{}-{}.p'.format(timestamp, args.mode, args.model), 'wb') as fp:
    pickle.dump(portfolio_value, fp)

  with open('portfolio_hist/{}-{}-{}.p'.format(timestamp, args.mode, args.model), 'wb') as fp:
    pickle.dump(env.portfolio_history, fp)

