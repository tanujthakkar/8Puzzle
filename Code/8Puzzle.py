#!/usr/env/bin python3

"""
ENPM661 Spring 2022: Planning for Autonomous Robots
Project 1: 8 Puzzel Problem

Author(s):
Tanuj Thakkar (tanuj@umd.edu)
M. Engg Robotics
University of Maryland, College Park
"""

# Importing modules
import sys
import os
import numpy as np
import argparse
from multiprocessing import Queue


class Node():

    def __init__(self, state: np.array, index: int, parent_index: int, actions: np.array) -> None:
        self.state = state
        self.index = index
        self.parent_index = parent_index
        self.actions = actions


class EightPuzzle():

    def __init__(self, initial_state: list(), goal_state: list()) -> None:
        self.initial_state = np.uint8(initial_state).reshape(3,3)
        self.goal_state = np.uint8(goal_state).reshape(3,3)
        self.actions = np.array([[0, 1],
                                 [0, -1],
                                 [-1, 0],
                                 [1, 0]]) # Action Set - RIGHT, LEFT, UP, DOWN
        self.current_index = 0
        self.initial_node = Node(self.initial_state, self.current_index, None, self.actions)
        self.goal_node = Node(self.goal_state, -1, None, None)

        print("Created a 8 Puzzle with\n")
        print("Initial State: ", initial_state)
        print("Goal State: ", goal_state)

    def find_zero(self, state):
        return np.array(np.where(state==0)).flatten()

    def is_valid_state(self, pos):
        if(pos[0] > 0 and pos[0] < self.state.shape[0] and pos[1] > 0 and pos[1] < self.state.shape[1]):
            return True
        else:
            return False

    def solve(self):

        q = Queue() # Using a queue for Breadth First Search

        q.put(self.initial_node) # Pushing the current node/inital node

        while(q.qsize() != 0):

            current_node = q.get()
            print(current_node.state)

            if((current_node == goal_node).all()):
                print("Goal Reached!")

            pos = self.find_zero(current_node.state)

            for action in current_node.actions:
                new_pos = pos + action
                new_state = np.copy(current_node.state)
                if(is_valid_state(new_state)):
                    new_state[tuple(pos)], new_state[tuple(new_pos)] = new_state[tuple(new_pos)], new_state[tuple(pos)]
                    new_index = current_index + 1
                    new_node = Node(new_state, new_index, current_node.index, )
                    print("State: ", new_state)
                    input('q')
                else:
                    print("Invalid state: ", new_state)


def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--InitialState', type=str, default="[[1,2,3],[4,0,5],[6,7,8]]", help='Initial state of the 8 Puzzle problem')
    Parser.add_argument('--GoalState', type=str, default="[[1,2,3][4,5,6][7,8,0]]", help='Goal state of the 8 Puzzle problem')  

    Args = Parser.parse_args()
    InitialState = Args.InitialState.replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
    GoalState = Args.GoalState.replace('[', ' ').replace(']', ' ').replace(',', ' ').split()

    EP = EightPuzzle(InitialState, GoalState)
    EP.solve()


if __name__ == '__main__':
    main()