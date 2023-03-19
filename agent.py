import torch
import random
import numpy as np
from collections import deque
from snakegame import AiSnakeGame, Direction, Point

MAX_MEMORY = 200_000
BATCH_SIZE = 2000
LR = 0.001

class Agent:
  def __init__(self):
    self.n_games = 0
    self.epsilon = 0 # randomness
    self.gamma = 0 # discount rate
    self.memory = deque(maxlen=MAX_MEMORY) # popleft()
    self.model = None # TODO
    self.trainer = None # TODO

  def get_state(self, game):
    head = game.snake[0]
    point_l = Point(head.x - 10, head.y)
    point_r = Point(head.x + 10, head.y)
    point_u = Point(head.x, head.y - 10)
    point_d = Point(head.x, head.y + 10)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
      # Danger ahead
      (dir_r and game.is_collision(point_r)) or
      (dir_l and game.is_collision(point_l)) or
      (dir_u and game.is_collision(point_u)) or
      (dir_d and game.is_collision(point_d)),

      # Danger right
      (dir_u and game.is_collision(point_r)) or
      (dir_d and game.is_collision(point_l)) or
      (dir_l and game.is_collision(point_u)) or
      (dir_r and game.is_collision(point_d)),

      # Danger left
      (dir_d and game.is_collision(point_r)) or
      (dir_u and game.is_collision(point_l)) or
      (dir_r and game.is_collision(point_u)) or
      (dir_l and game.is_collision(point_d)),

      # Movement direction
      dir_l,
      dir_r,
      dir_u,
      dir_d,

      # Food location
      game.food.x < game.head.x, # food on left side
      game.food.x > game.head.x, # food on the right side
      game.food.y < game.head.y,  # food on the up side
      game.food.y > game.head.y  # food on the down side
    ]
    return np.array(state, dtype=int)


  def remeber(self, state, action, reward, next_state, game_over):    
    self.memory.append((state, action, reward, next_state, game_over)) #popleft if overflows the max memory
  
  def train_long_memo(self):
    if len(self.memory) > BATCH_SIZE:
      mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
    else: 
      mini_sample = self.memory

    states, actions, rewards, next_states, games_over = zip(*mini_sample)
    self.trainer.train_step(states, actions, rewards, next_states, games_over)

  def train_short_memo(self, state, action, reward, next_state, game_over):
    self.trainer.train_step(state, action, reward, next_state, game_over)
  
  def get_action(self, state):
    # random moves: tradeoffs exploration
    self.epsilon = 80 - self.n_games
    final_move = [0,0,0]
    if random.randint(0,200) < self.epsilon:
      move = random.randint(0,2)
      final_move[move] = 1
    else:
      state0 = torch.tensor(state, dtype=torch.float)
      prediction = self.model.predict(state0)
      move = torch.argmax(prediction).item()
      final_move[move] = 1
    return final_move
  
def train():
  plot_scores = []
  plot_mean_score = []
  total_score = 0
  best_score = 0
  agent = Agent()
  game = AiSnakeGame()
  while True:
    # get old state
    state_old = agent.get_state(game)

    #get move
    final_move = agent.get_action(state_old)

    # make the move and get new state
    reward, game_over, score = game.play_step(final_move)
    new_state = agent.get_state(game)

    # train to short memo
    agent.train_short_memo(state_old, final_move, reward, new_state, game_over)

    # memorize
    agent.remeber(state_old, final_move, reward, new_state, game_over)

    if game_over:
      # train long memo, plot result
      game.reset()
      agent.n_game += 1
      agent.train_long_memo
      if score > best_score:
        best_score = score
        # agent.model.save()

      print("game", agent.n_games, 'Score', score, 'Record: ', best_score)
      # TODO plot


if __name__ == '__main__':
   train()