# coding: utf-8

import numpy as np
import copy
import sys

from Game import Game
from GameConfig import *
from Network import Network

class HumanVSAI:
    def __init__(self, model_file=None):
        self.network = Network(BOARD_WIDTH, BOARD_HEIGHT, len(SHIPS))
        if not model_file is None:
            self.network.restoreModel('./models/mymodel')

        self.ai_game = Game(BOARD_WIDTH, BOARD_HEIGHT, SHIPS, network=self.network)
        self.human_game = Game(BOARD_WIDTH, BOARD_HEIGHT, SHIPS)

    def playOneGame(self):
        ai_input_dimensions = self.ai_game.board.getInputDimensions()
        human_input_dimensions = self.human_game.board.getInputDimensions()
        cur_step = 0
        while True:

            if ai_input_dimensions is None:
                print('LOL AI wins !')
                sys.exit()
            if human_input_dimensions is None:
                print('You win !')
                sys.exit()

            print('Current Step: ' + str(cur_step))
            self.printBothBoards()

            ai_available_moves = self.ai_game.board.getNextAvailableBombLocations()
            ai_next_move = self.ai_game.getBestMoveBasedOnModel(ai_input_dimensions, ai_available_moves)
            (ai_input_dimensions, _, _) = self.ai_game.takeAMove(ai_next_move)

            human_available_moves = self.human_game.board.getNextAvailableBombLocations()
            human_next_move = self.getHumanMoveInput()
            if human_next_move == None:
                human_next_move = self.human_game.getRandomMove(human_available_moves)
            (human_input_dimensions, _, _) = self.human_game.takeAMove(human_next_move)

            cur_step += 1

    def getHumanMoveInput(self):
        while True:
            human_move = input("Input next move (e.g. '2,4') or random if no input: ")
            if human_move == '':
                return None
            xy = human_move.split(',')
            if len(human_move) != 2:
                print('Invalid input!')
                continue
            x = human_move[0]
            y = human_move[1]
            location = x * self.human_game.board.board_width + y
            if self.human_game.board.available_moves[location] != 1:
                print('The location is already been taken!')
                continue
            return location

    def printBothBoards(self):
        ai_board_printer = self.ai_game.board.getViewState()
        human_board_printer = self.human_game.board.getViewState()
        spaces = '                      '
        for i in range(len(ai_board_printer)):
            print(ai_board_printer[i] + spaces + human_board_printer[i])

gamer = HumanVSAI('./models/mymodel')
gamer.playOneGame()
