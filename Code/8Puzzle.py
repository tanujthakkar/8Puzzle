#!/usr/env/bin python3

"""
ENPM661 Spring 2022: Planning for Autonomous Robots
Project 1: 8 Puzzle Problem

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
import time
from queue import Queue


class Node():
    '''
        Class to represent nodes in Breadth First Search

        Attributes
        state: state of the node
        index: index of the node
        parent_index: index of parent node
        actions: Possible actions to generate child nodes
    '''

    def __init__(self, state: np.array, index: int, parent_index: int, actions: np.array) -> None:
        self.state = state
        self.index = index
        self.parent_index = parent_index
        self.actions = actions


class EightPuzzle():
    '''
        Class for the 8 Puzzle Problem

        Attributes
        initial_state: Initial state of the puzzle
        goal_state: Goal state of the puzzle
        actions: All possible actions to generate child nodes
        current_index: Current index of the search
        valid: Boolean to verify validity of inital and goal states
    '''

    def __init__(self, initial_state: list(), goal_state: list()) -> None:
        self.initial_state = np.uint8(initial_state).reshape(3,3).transpose()
        self.goal_state = np.uint8(goal_state).reshape(3,3).transpose()
        self.valid = True
        if(not self.is_valid_input(self.initial_state)):
            print("INITIAL STATE INVALID!")
            self.valid = False
            return
        if(not self.is_valid_input(self.goal_state)):
            print("GOAL STATE INVALID!")
            self.valid = False
            return
        self.actions = np.array([[0, 1],
                                 [0, -1],
                                 [-1, 0],
                                 [1, 0]]) # Action Set - RIGHT, LEFT, UP, DOWN
        self.current_index = 0

        self.initial_node = Node(self.initial_state, self.current_index, 0, self.actions)
        self.goal_node = Node(self.goal_state, -1, None, None)
        self.visited_dict = dict()
        self.parent_index_dict = dict()
        self.final_node = None
        self.path = None

        print("\nCreated a 8 Puzzle with...\n")
        print("\nInitial State: \n", self.initial_node.state)
        print("\nGoal State: \n", self.goal_node.state)

    def is_valid_input(self, state: np.array) -> bool:
        valid = np.arange(9)
        s = state.flatten()
        for i in s:
            if(i not in valid or (s == i).sum() != 1):
                return False

        return True

    def find_zero(self, state: np.array) -> np.array:
        return np.array(np.where(state==0)).flatten()

    def is_valid_state(self, pos: np.array) -> bool:
        if(pos[0] >= 0 and pos[0] < self.initial_state.shape[0] and pos[1] >= 0 and pos[1] < self.initial_state.shape[1]):
            return True
        else:
            return False

    def to_tuple(self, state: np.array) -> tuple:
        return tuple(map(tuple, state))

    def is_solvable_state(self, state: np.array) -> bool:

        inversions = 0
        s = state.reshape(9)

        for i in range(0, 9):
            for j in range(i+1, 9):
                if(s[i] != 0 and s[j] != 0 and s[i] > s[j]):
                    inversions += 1

        return True if(inversions%2 == 0) else False

    def solve(self) -> bool:

        q = Queue() # Using a queue for Breadth First Search

        q.put(self.initial_node) # Pushing the current node/inital node

        print("\nStarting search...")

        s = set()

        tick = time.time()
        while(not q.empty()):

            current_node = q.get()
            s.add(self.to_tuple(current_node.state))
            self.visited_dict[current_node.index] = current_node

            if((current_node.state == self.goal_node.state).all()):
                print("GOAL REACHED!")
                toc = time.time()
                print("Took %.03f seconds to search the path"%((toc-tick)))
                self.final_node = current_node
                return True

            if(not self.is_solvable_state(current_node.state)):
                continue

            pos = self.find_zero(current_node.state)

            for action in range(len(current_node.actions)):
                new_pos = pos + current_node.actions[action]
                new_state = np.copy(current_node.state)
                if(self.is_valid_state(new_pos)):
                    new_state[tuple(pos)], new_state[tuple(new_pos)] = new_state[tuple(new_pos)], new_state[tuple(pos)]
                    new_index = self.current_index + 1    
                    self.current_index = new_index
                    new_action_set = np.delete(np.copy(self.actions), action, axis=0)
                    new_node = Node(new_state, new_index, current_node.index, self.actions)

                    if(self.to_tuple(new_state) in s):
                        self.current_index -= 1
                        continue
                    
                    q.put(new_node)

                else:
                    # print("Invalid state: \n", new_pos, new_state)
                    pass

        return False

    def generate_path(self) -> list():

        current_node = self.final_node
        self.path = list()

        print("BACKTRACKING PATH...")

        while(current_node.index != 0):
            self.path.append(current_node.state.transpose())
            current_node = self.visited_dict[current_node.parent_index]

        self.path.append(self.initial_node.state.transpose())
        self.path.reverse()

        # for node in self.path:
        #     print(node)

        print("BACKTRACKING PATH COMPLETE!")
        return self.path

    def generate_data_files(self) -> None:

        print("GENERATING DATA FILES...")

        # Generating nodesPath.txt
        path = np.asarray(self.path).reshape(-1, 9)
        np.savetxt('nodePath.txt', path, delimiter=' ', fmt='%d')

        # Generating NodesInfo.txt
        indexes = [[i, self.visited_dict[i].parent_index, 0] for i in self.visited_dict]
        # indexes = np.c_[indexes, np.zeros(len(indexes))]
        np.savetxt('NodesInfo.txt', indexes, header="Node_index Parent_Node_index Cost", comments='', delimiter=' ', fmt='%d')

        # Generating Nodes.txt
        visited_states = [(self.visited_dict[i].state.transpose()).reshape(9) for i in self.visited_dict]
        np.savetxt('Nodes.txt', visited_states, delimiter=' ', fmt='%d')

        print("DATA FILES GENERATED!")


def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--InitialState', type=str, default="[1,4,7],[5,0,8],[2,3,6]", help='Initial state of the 8 Puzzle problem [column-wise]')
    Parser.add_argument('--GoalState', type=str, default="[1,4,7],[2,5,8],[3,6,0]", help='Goal state of the 8 Puzzle problem [column-wise]')

    Args = Parser.parse_args()
    InitialState = Args.InitialState.replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
    GoalState = Args.GoalState.replace('[', ' ').replace(']', ' ').replace(',', ' ').split()

    EP = EightPuzzle(InitialState, GoalState)
    if(EP.valid):
        if(EP.solve()):
            EP.generate_path()
            EP.generate_data_files()
        else:
            print("Goal state is UNRECHEABLE!")

if __name__ == '__main__':
    main()