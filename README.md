# Snake game IA Python
## install anaconda
To create an enviroment and install the tools to build our game, you need to install conda. Donwload it from the [`oficial website`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

## Creating an env
  To create an env to run our code, run this on your command lime: 
  ```bash
  $ conda create -n py_game_env python=3.11
  ```
  To access it, just run:
  ```bash
  $ conda activate py_game_env
  ```

## Installing dependencies
  To install the dependencies of our game, first make sure you're on the python enviroment that we created on conda
  ### Pygame
  To install pygame lib, run: 
  ```bash
    $ pip install pygame
  ```
  ### Torch
  This lib we will use for the neural network, pytorch is a great and easy to learn python library.
  ```bash
  $ pip install torch torchvision
  ```
  ### Matplotlib
  We will need it later on the code
  ```bash
    $ pip install matplotlib ipython
  ```

## Running our game
  To run the snake game, first make sure you're on the python enviroment that we created previously on conda.

  ### Runing it
  ```bash
    $ python3 snake-game.py
  ```
