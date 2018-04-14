# coding: utf-8
# Reference: http://efavdb.com/battleship/

from Game import Game
from GameConfig import *
from Network import Network

class TrainGame:
    def __init__(self, model_file=None):
        self.gamma = 0.5
        self.alpha = 0.01

        self.network = Network(BOARD_WIDTH, BOARD_HEIGHT, len(SHIPS))
        if not model_file is None:
            self.network.restoreModel('./models/mymodel')

        self.game = Game(BOARD_WIDTH, BOARD_HEIGHT, SHIPS, network=self.network)
        self.total_ships_lengths = sum([ship['length'] for ship in self.game.board.ships])
        self.board_size = self.game.board.board_height * self.game.board.board_width
        self.max_train_step = 300000

    def selfPlayOneGame(self):
        all_input_states = []
        all_moves = []
        all_hits = []
        self.game.resetBoard()
        (input_dimensions, move, is_hit) = self.game.takeAMove()
        while not input_dimensions is None:
            all_input_states.append(input_dimensions)
            all_moves.append(move)
            all_hits.append(is_hit)
            (input_dimensions, move, is_hit) = self.game.takeAMove()

        all_discounted_reward = self.rewardsCalculator(all_hits)
        return (all_input_states, all_moves, all_hits, all_discounted_reward)

    def rewardsCalculator(self, hit_log, gamma=0.5):
        """ Discounted sum of future hits over trajectory"""
        hit_log_weighted = [(item -
            float(self.total_ships_lengths - sum(hit_log[:index])) / float(self.board_size - index)) *
            (gamma ** index) for index, item in enumerate(hit_log)]
        return [((gamma) ** (-i)) * sum(hit_log_weighted[i:]) for i in range(len(hit_log))]

    def trainWithSelfPlay(self):
        all_entropy = 0
        all_num_move = 0
        batch_size = 50
        for i in range(self.max_train_step):
            (all_input_states, all_moves, all_hits, all_discounted_reward) = self.selfPlayOneGame()
            all_num_move += len(all_hits)
            for input_states, moves, discounted_reward in zip(all_input_states, all_moves, all_discounted_reward):
                entropy = self.network.runTrainStep(input_states, [moves], self.alpha * discounted_reward)
            if i % batch_size == 0 and i != 0:
                print('Total Game: ' + str(i) + ' ' + ' Avg moves: ' + str(all_num_move * 1.0/ batch_size * 1.0))
                all_entropy = 0
                all_num_move = 0

                if i % (batch_size * 20) == 0:
                    self.game.network.saveModel('./models/mymodel')
                    print("SAVE MODEL # {}".format(i))

train_game = TrainGame('./models/mymodel')#model_file='./models/mymodel')
train_game.trainWithSelfPlay()
