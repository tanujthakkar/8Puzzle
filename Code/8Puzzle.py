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
from pprint import pprint
import time
from queue import Queue


class Node():

    def __init__(self, state: np.array, index: int, parent_index: int, actions: np.array) -> None:
        self.state = state
        self.index = index
        self.parent_index = parent_index
        self.actions = actions

    def node_summary(self):
        pprint(self)


class EightPuzzle():

    def __init__(self, initial_state: list(), goal_state: list()) -> None:
        self.initial_state = np.uint8(initial_state).reshape(3,3)
        self.goal_state = np.uint8(goal_state).reshape(3,3)
        self.actions = np.array([[0, 1],
                                 [0, -1],
                                 [-1, 0],
                                 [1, 0]]) # Action Set - RIGHT, LEFT, UP, DOWN
        self.current_index = 0
        self.initial_node = Node(self.initial_state, self.current_index, 0, self.actions)
        self.goal_node = Node(self.goal_state, -1, None, None)
        self.closed_dict = None
        self.parent_index_dict = None
        self.final_node = None
        self.path = None

        print("Created a 8 Puzzle with\n")
        print("Initial State: ", vars(self.initial_node))
        print("Goal State: ", vars(self.goal_node))

    def find_zero(self, state) -> np.array:
        return np.array(np.where(state==0)).flatten()

    def is_valid_state(self, pos) -> bool:
        if(pos[0] >= 0 and pos[0] < self.initial_state.shape[0] and pos[1] >= 0 and pos[1] < self.initial_state.shape[1]):
            return True
        else:
            return False

    def calc_index(self, pos:) -> int:
        return int(pos[0] * self.initial_state.shape[0] + pos[1])

    def to_tuple(self, state: np.array) -> tuple:
        return tuple(map(tuple, state))

    def solve(self) -> bool:

        q = Queue() # Using a queue for Breadth First Search

        q.put(self.initial_node) # Pushing the current node/inital node

        open_set = dict()
        closed_dict = dict()
        parent_index_dict = dict()

        open_set[self.initial_node.index] = self.initial_node.state

        tick = time.time()
        while(q.qsize() != 0):

            # if(self.current_index%1000):
                # print(self.current_index)

            # if(((time.time()-tick)/60) > 10.0):
            #     print("Time Limit Exceeded!")
            #     return

            current_node = q.get()
            closed_dict[self.to_tuple(current_node.state)] = current_node.index
            parent_index_dict[current_node.index] = current_node.parent_index
            # closed_dict.add(self.to_tuple(current_node.state))
            # print(self.to_tuple(current_node.state))
            # input('q')
            # open_set.pop(current_node.index)
            # print(current_node.state)

            if((current_node.state == self.goal_node.state).all()):
                pprint(vars(current_node))
                print("Goal Reached!")
                toc = time.time()
                print("Took %.03f seconds to train"%((toc-tick)))
                self.closed_dict = closed_dict
                self.final_node = current_node
                return True

            pos = self.find_zero(current_node.state)

            for action in range(len(current_node.actions)):
                new_pos = pos + current_node.actions[action]
                new_state = np.copy(current_node.state)
                if(self.is_valid_state(new_pos)):
                    new_state[tuple(pos)], new_state[tuple(new_pos)] = new_state[tuple(new_pos)], new_state[tuple(pos)]
                    new_index = self.current_index + 1
                    self.current_index = new_index
                    new_action_set = np.delete(np.copy(self.actions), action, axis=0)
                    # print(new_action_set)
                    # print(new_action_set.reshape(len(new_action_set)//2, 2))
                    new_node = Node(new_state, new_index, current_node.index, self.actions)

                    if(self.to_tuple(new_state) in closed_dict):
                        # print("In closed_dict")
                        continue

                    # if(new_index not in open_set):
                    #     print("In open_set")
                    #     open_set[new_index] = new_state
                    
                    q.put(new_node)

                    if((new_node.state == self.goal_node.state).all()):
                        closed_dict[self.to_tuple(new_node.state)] = new_node.index
                        parent_index_dict[new_node.index] = new_node.parent_index
                        pprint(vars(new_node))
                        print("Goal Node Created!")
                        toc = time.time()
                        print("Took %.03f seconds to train"%((toc-tick)))
                        self.closed_dict = closed_dict
                        self.final_node = new_node
                        self.parent_index_dict = parent_index_dict
                        return True

                    # pprint(vars(new_node))
                    # input('q')
                else:
                    pass
                    # print("Invalid state: \n", new_pos, new_state)

    def backtrack_path(self) -> list():

        current_state = self.to_tuple(self.final_node.state)
        current_index = self.final_node.index
        path = list()

        visited_state_list = list(self.closed_dict.keys())
        visited_index_list = list(self.closed_dict.values())

        while(current_index != 0):

            path.append(current_state)
            print(path[-1])
            current_index = self.closed_dict[current_state]
            current_state = visited_state_list[visited_index_list.index(self.parent_index_dict[current_index])]

        print("Backtracking finished...")

        path.reverse()

        for node in path:
            print(np.asarray(node))

        self.path = path

        return self.path


def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--InitialState', type=str, default="[[0,1,2],[3,4,5],[6,7,8]]", help='Initial state of the 8 Puzzle problem')
    Parser.add_argument('--GoalState', type=str, default="[[1,2,3],[4,5,6],[7,8,0]]", help='Goal state of the 8 Puzzle problem')  

    Args = Parser.parse_args()
    InitialState = Args.InitialState.replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
    GoalState = Args.GoalState.replace('[', ' ').replace(']', ' ').replace(',', ' ').split()

    EP = EightPuzzle(InitialState, GoalState)
    EP.solve()
    EP.backtrack_path()


if __name__ == '__main__':
    main()