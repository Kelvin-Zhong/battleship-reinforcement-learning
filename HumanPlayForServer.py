# coding: utf-8

import numpy as np
import copy
import sys
import pickle
import os.path

from Game import Game
from GameConfig import *
from Network import Network

class HumanVSAIForServer:
    def __init__(self, model_file=None, human_name=None):
        self.human_file = human_name + '.pickle'
        self.ai_file = human_name + '_ai.pickle'
        ai_board, human_board = self.resumeBoards()

        self.network = Network(BOARD_WIDTH, BOARD_HEIGHT, len(SHIPS))
        if not model_file is None:
            self.network.restoreModel('./models/mymodel')

        self.ai_game = Game(BOARD_WIDTH, BOARD_HEIGHT, SHIPS, board=ai_board, network=self.network)
        self.human_game = Game(BOARD_WIDTH, BOARD_HEIGHT, SHIPS, board=human_board)
        self.cur_step = 0

    def getBothBoardsString(self):
        ai_board_printer = self.ai_game.board.getViewState()
        human_board_printer = self.human_game.board.getViewState()
        spaces = '   '
        output_board = []
        for i in range(len(ai_board_printer)):
            output_board.append(ai_board_printer[i] + spaces + human_board_printer[i])

        # for i in range(len(output_board)):
        #     print(output_board[i])

        output_string = '        AI                   YOU  \n'
        for i in range(len(output_board)):
            output_string += output_board[i] + '\n'

        return output_string

    def getGameStateString(self):
        state_number = self.ai_game.board.state_number
        ai_hits = 0
        for i in range(len(state_number)):
            for j in range(len(state_number[0])):
                if state_number[i][j] == 1:
                    ai_hits += 1

        human_board_printer = self.human_game.board.getViewState()
        output_board = ['AI hit: ' + str(ai_hits)]
        for i in range(len(human_board_printer)):
            output_board.append(human_board_printer[i])

        output_board = 'AI hit: ' + str(ai_hits) + '\n'
        for i in range(len(human_board_printer)):
            output_board += human_board_printer[i] + '\n'
        return output_board

    def takeOneMove(self, human_move):
        human_next_move = self.getHumanMoveInput(human_move)
        if human_next_move is None:
            return (None, self.getGameStateString() + 'Invalid Input, please switch to English input and input the following format: row,column, e.g. 2,4\n Or you can input "reset" to reset the game')

        ai_input_dimensions = self.ai_game.board.getInputDimensions()
        human_input_dimensions = self.human_game.board.getInputDimensions()
        ai_available_moves = self.ai_game.board.getNextAvailableBombLocations()
        ai_next_move = self.ai_game.getBestMoveBasedOnModel(ai_input_dimensions, ai_available_moves)
        # print(ai_next_move)
        (ai_input_dimensions, _, _) = self.ai_game.takeAMove(ai_next_move)

        (human_input_dimensions, _, _) = self.human_game.takeAMove(human_next_move)

        self.cur_step += 1

        is_ai_win = self.ai_game.board.checkIfGameFinished()
        is_human_win = self.human_game.board.checkIfGameFinished()
        if is_ai_win:
            self.deleteFiles()
            return (None, self.getBothBoardsString() + 'LOL AI wins !')

        if is_human_win:
            self.deleteFiles()
            return (None, self.getBothBoardsString() + 'You win !')

        # Save the game
        self.saveBoards()

        return (1, self.getGameStateString())

    def getHumanMoveInput(self, human_move):
        if human_move == '':
            return None
        xy = human_move.split(',')
        if len(xy) != 2:
            return None
        x = int(xy[0])
        y = int(xy[1])
        if x >= self.human_game.board.board_height or x < 0:
            return None
        if y >= self.human_game.board.board_width or y < 0:
            return None
        location = x * self.human_game.board.board_width + y
        if self.human_game.board.available_bomb_locations[location] != 1:
            return None
        return location

    def saveBoards(self):
        with open(self.ai_file, 'wb') as ai_f:
            pickle.dump(self.ai_game.board, ai_f, pickle.HIGHEST_PROTOCOL)
        with open(self.human_file, 'wb') as human_f:
            pickle.dump(self.human_game.board, human_f, pickle.HIGHEST_PROTOCOL)

    # Resume game, or set up a new game if not found
    def resumeBoards(self):
        ai_board = None
        human_board = None

        if not os.path.isfile(self.human_file) or not os.path.isfile(self.ai_file):
            return (ai_board, human_board)

        with open(self.ai_file, 'rb') as ai_f:
            ai_board = pickle.load(ai_f)
        with open(self.human_file, 'rb') as human_f:
            human_board = pickle.load(human_f)
        return (ai_board, human_board)

    def deleteFiles(self):
        os.remove(self.human_file)
        os.remove(self.ai_file)

# my_name = 'tmp'
# gamer = HumanVSAIForServer(human_name=my_name)
# gamer.takeOneMove('0,1')
