# coding: utf-8

import copy
import random
import numpy as np

class Board:
    def __init__(self, board_height, board_width, ships):
        self.board_height = board_height
        self.board_width = board_width
        self.state_number = [[0 for i in range(self.board_width)] for j in range(self.board_height)]
        self.true_state = [['-' for i in range(self.board_width)] for j in range(self.board_height)]
        self.view_state = [['-' for i in range(self.board_width)] for j in range(self.board_height)]
        self.ships = copy.deepcopy(ships)
        for i in range(len(self.ships)):
            self.ships[i]['remaining_length'] = self.ships[i]['length']
        self.remaining_ships = copy.deepcopy(ships)
        # index start at 1
        self.available_bomb_locations = np.full(self.board_height * self.board_width, 1, 'float32')

        # random place the ships
        self.randomPlacement()

    def printStateNumber(self):
        for i in range(self.board_height):
            row = ''
            for j in range(self.board_width):
                row += str(self.state_number[i][j]) + ' '
            print(row)

    def printTrueState(self):
        for i in range(self.board_height):
            row = ''
            for j in range(self.board_width):
                row += self.true_state[i][j] + ' '
            print(row)

    def getViewState(self):
        statePrinter = []
        row = '  '
        for j in range(self.board_width):
            row += str(j) + ' '
        statePrinter.append(row)

        for i in range(self.board_height):
            row = str(i) + ' '
            for j in range(self.board_width):
                row += self.view_state[i][j] + ' '
            statePrinter.append(row)
        return statePrinter

    def getNextShipAvailablePlacements(self):
        if len(self.remaining_ships) == 0:
            return (None, None)
        cur_ship = self.remaining_ships.pop(0)
        ship_length = cur_ship['length']
        available_placement = []

        for i in range(self.board_height):
            for j in range(self.board_width):
                # check horizontal line
                eligible_flag = True
                for k in range(ship_length):
                    if j + k >= self.board_width or self.true_state[i][j + k] != '-':
                        eligible_flag = False
                        break
                if eligible_flag:
                    available_placement.append({'x': i, 'y': j, 'z': 0})

                # check vertical line
                eligible_flag = True
                for k in range(ship_length):
                    if i + k >= self.board_height or self.true_state[i + k][j] != '-':
                        eligible_flag = False
                        break
                if eligible_flag:
                    available_placement.append({'x': i, 'y': j, 'z': 1})

        return (cur_ship, available_placement)

    def placeShip(self, ship, placement):
        x = placement['x']
        y = placement['y']
        z = placement['z']
        ship_length = ship['length']
        ship_mark = ship['mark']

        if z == 0:
            for i in range(ship_length):
                self.true_state[x][y + i] = ship_mark
        else:
            for i in range(ship_length):
                self.true_state[x + i][y] = ship_mark

    def checkIfGameFinished(self):
        for ship in self.ships:
            if not ship['remaining_length'] == 0:
                return False
        return True

    def getNextAvailableBombLocations(self):
        return copy.deepcopy(self.available_bomb_locations)

    def placeBombAndCheckIfHit(self, location):
        location = int(location)
        self.available_bomb_locations[location] = 0
        x, y = divmod(location, self.board_width)

        is_hit = 0
        if self.true_state[x][y] != '-':
            self.state_number[x][y] = 1
            for i in range(len(self.ships)):
                # If hits a ship
                ship_mark = self.ships[i]['mark']
                if ship_mark == self.true_state[x][y]:
                    self.ships[i]['remaining_length'] -= 1
                    is_hit = 1

                    if self.ships[i]['remaining_length'] == 0:
                        for j in range(self.board_height):
                            for k in range(self.board_width):
                                if self.true_state[j][k] == ship_mark:
                                    self.view_state[j][k] = ship_mark
                    else:
                        self.view_state[x][y] = 'O'

        else:
            self.state_number[x][y] = -1
            self.view_state[x][y] = 'X'

        return is_hit

    '''
    Get different input dimensions based on current state:
    (each dimension is a 2D array with board_height * board_width)
    1. board state to determine which place is bombed
    2. Ship states: for each ship, use the flag showing if the ship is sink or not
    '''
    def getInputDimensions(self):
        input_dimensions = np.array([self.state_number], dtype='float32').flatten()
        for i in range(len(self.ships)):
            sink_flag = 0 if self.ships[i]['remaining_length'] == 0 else 1
            ship_dimension = np.full((self.board_height, self.board_width), sink_flag, dtype='float32').flatten()
            input_dimensions = np.append(input_dimensions, ship_dimension)

        input_dimensions = input_dimensions.flatten().reshape(1, -1)
        return input_dimensions

    def randomPlacement(self):
        while True:
            ship, available_placements = self.getNextShipAvailablePlacements()
            if ship == None:
                break

            placement = random.choice(available_placements)
            self.placeShip(ship, placement)

class Game:
    def __init__(self, board_height, board_width, ships, board=None, network=None):
        self.network = network
        self.board_height = board_height
        self.board_width = board_width
        self.ships = ships
        if board is None:
            self.board = Board(self.board_height, self.board_width, self.ships)
        else:
            self.board = board

    def resetBoard(self):
        self.board = Board(self.board_height, self.board_width, self.ships)

    def takeAMove(self, next_move=None):
        if self.board.checkIfGameFinished():
            return (None, None, None)

        input_dimensions = self.board.getInputDimensions()
        if next_move is None:
            available_moves = self.board.getNextAvailableBombLocations()
            next_move = self.getBestMoveBasedOnModel(input_dimensions, available_moves)

        is_hit = self.board.placeBombAndCheckIfHit(next_move)
        return (input_dimensions, next_move, is_hit)

    def getBestMoveBasedOnModel(self, input_dimensions, available_moves):
        new_shape = list(np.array(input_dimensions).shape)
        input_dimensions = np.reshape(input_dimensions, new_shape)
        board_probs = self.network.getBoardProbabilities(input_dimensions)
        board_probs_max_index = np.argmax(board_probs)
        board_probs = np.multiply(board_probs, available_moves)
        next_move = np.argmax(board_probs)
        return next_move

    def getRandomMove(self, available_moves):
        available_moves_p = copy.deepcopy(available_moves)
        num_available_moves = np.sum(available_moves)
        available_moves_p[available_moves_p != 1] = 0
        available_moves_p[available_moves_p == 1] = 1.0 / num_available_moves
        next_move = np.random.choice(np.arange(len(available_moves)), p=available_moves_p)
        return next_move
