from typing import List, Tuple, Union
import numpy as np
import torch.nn as nn

from utils.pytorch_models import ResnetModel
from .environment_abstract import Environment, State
from random import randrange
import random


class BloxorzState(State):
    __slots__ = ['block_pos', 'orientation', 'hash']

    def __init__(self, block_pos: np.ndarray, orientation: int):
        self.block_pos: np.ndarray = block_pos
        self.orientation: int = orientation
        self.hash = None

    def __hash__(self):
        if self.hash is None:
            self.hash = hash((self.block_pos[0], self.block_pos[1], self.orientation))
        return self.hash

    def __eq__(self, other):
        return (np.array_equal(self.block_pos, other.block_pos) and 
                self.orientation == other.orientation)


class Bloxorz(Environment):
    moves: List[str] = ['U', 'D', 'L', 'R']
    moves_rev: List[str] = ['D', 'U', 'R', 'L']

    def __init__(self, grid_size: Tuple[int, int] = (8, 8)):
        super().__init__()
        
        self.grid_size: Tuple[int, int] = grid_size
        self.dtype = np.uint8
        
        # Goal position - make it reachable
        self.goal_pos: np.ndarray = np.array([grid_size[0] - 2, grid_size[1] - 2])
        
        # Generate obstacles that create a solvable path
        self.obstacles: List[Tuple[int, int]] = self._generate_obstacles()
        
        # Weak tiles
        self.weak_tiles: List[Tuple[int, int]] = [(3, 3), (3, 4), (4, 3), (4, 4)]

    def next_state(self, states: List[BloxorzState], action: int) -> Tuple[List[BloxorzState], List[float]]:
        states_next: List[BloxorzState] = []
        transition_costs: List[float] = []
        
        for state in states:
            new_pos, new_orientation = self._apply_move(state.block_pos, state.orientation, action)
            
            if self._is_valid_position(new_pos, new_orientation):
                states_next.append(BloxorzState(new_pos, new_orientation))
                transition_costs.append(1.0)
            else:
                # Stay in current state with penalty
                states_next.append(BloxorzState(state.block_pos.copy(), state.orientation))
                transition_costs.append(10.0)  # Higher penalty for invalid moves
        
        return states_next, transition_costs

    def prev_state(self, states: List[BloxorzState], action: int) -> List[BloxorzState]:
        # For backwards moves, we need to be more careful about validity
        states_prev: List[BloxorzState] = []
        
        for state in states:
            # Try all possible reverse moves to find valid previous states
            valid_prev_found = False
            for test_action in range(self.get_num_moves()):
                test_pos, test_orientation = self._apply_move(state.block_pos, state.orientation, test_action)
                if self._is_valid_position(test_pos, test_orientation):
                    # Check if this move would lead to current state
                    reverse_pos, reverse_orientation = self._apply_move(test_pos, test_orientation, action)
                    if (np.array_equal(reverse_pos, state.block_pos) and 
                        reverse_orientation == state.orientation):
                        states_prev.append(BloxorzState(test_pos, test_orientation))
                        valid_prev_found = True
                        break
            
            if not valid_prev_found:
                # If no valid reverse move, just use the direct reverse
                move_rev_idx = self.moves.index(self.moves_rev[action])
                prev_pos, prev_orientation = self._apply_move(state.block_pos, state.orientation, move_rev_idx)
                states_prev.append(BloxorzState(prev_pos, prev_orientation))
        
        return states_prev

    def generate_goal_states(self, num_states: int, np_format: bool = False) -> Union[List[BloxorzState], np.ndarray]:
        goal_states = [BloxorzState(self.goal_pos.copy(), 0) for _ in range(num_states)]
        
        if np_format:
            return self.state_to_nnet_input(goal_states)[0]
        else:
            return goal_states

    def is_solved(self, states: List[BloxorzState]) -> np.ndarray:
        solved = np.zeros(len(states), dtype=bool)
        for i, state in enumerate(states):
            solved[i] = (state.orientation == 0 and 
                        state.block_pos[0] == self.goal_pos[0] and 
                        state.block_pos[1] == self.goal_pos[1])
        return solved

    def state_to_nnet_input(self, states: List[BloxorzState]) -> List[np.ndarray]:
        # More efficient representation: [x, y, orientation, goal_x, goal_y]
        states_np = np.zeros((len(states), 5), dtype=self.dtype)
        
        for i, state in enumerate(states):
            states_np[i, 0] = state.block_pos[0]  # x
            states_np[i, 1] = state.block_pos[1]  # y
            states_np[i, 2] = state.orientation   # orientation
            states_np[i, 3] = self.goal_pos[0]    # goal x
            states_np[i, 4] = self.goal_pos[1]    # goal y
        
        return [states_np]

    def get_num_moves(self) -> int:
        return len(self.moves)

    def get_nnet_model(self) -> nn.Module:
        # Input dimension is 5: [x, y, orientation, goal_x, goal_y]
        state_dim = 5
        nnet = ResnetModel(state_dim, self.grid_size[0] * self.grid_size[1], 5000, 1000, 4, 1, True)
        return nnet

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[BloxorzState], List[int]]:
        # Start from goal and apply random valid moves backwards
        states: List[BloxorzState] = self.generate_goal_states(num_states)
        scramble_nums: List[int] = np.random.randint(backwards_range[0], backwards_range[1] + 1, num_states).tolist()
        
        for i in range(num_states):
            current_state = states[i]
            scrambles_done = 0
            max_attempts = scramble_nums[i] * 10  # Prevent infinite loops
            
            while scrambles_done < scramble_nums[i] and max_attempts > 0:
                move = randrange(self.get_num_moves())
                new_states = self.prev_state([current_state], move)
                
                # Only accept if the new state is different and valid
                if (new_states[0] != current_state and 
                    self._is_valid_position(new_states[0].block_pos, new_states[0].orientation)):
                    current_state = new_states[0]
                    scrambles_done += 1
                
                max_attempts -= 1
            
            states[i] = current_state
        
        return states, scramble_nums

    def _apply_move(self, pos: np.ndarray, orientation: int, action: int) -> Tuple[np.ndarray, int]:
        new_pos = pos.copy()
        new_orientation = orientation
        
        move = self.moves[action]
        
        if orientation == 0:  # Standing
            if move == 'U':
                new_pos[1] -= 2
                new_orientation = 2
            elif move == 'D':
                new_pos[1] += 1
                new_orientation = 2
            elif move == 'L':
                new_pos[0] -= 2
                new_orientation = 1
            elif move == 'R':
                new_pos[0] += 1
                new_orientation = 1
                
        elif orientation == 1:  # Horizontal
            if move == 'U':
                new_pos[1] -= 1
            elif move == 'D':
                new_pos[1] += 1
            elif move == 'L':
                new_pos[0] -= 1
                new_orientation = 0
            elif move == 'R':
                new_pos[0] += 2
                new_orientation = 0
                
        elif orientation == 2:  # Vertical
            if move == 'U':
                new_pos[1] -= 1
                new_orientation = 0
            elif move == 'D':
                new_pos[1] += 2
                new_orientation = 0
            elif move == 'L':
                new_pos[0] -= 1
            elif move == 'R':
                new_pos[0] += 1
        
        return new_pos, new_orientation

    def _is_valid_position(self, pos: np.ndarray, orientation: int) -> bool:
        x, y = pos
        
        if orientation == 0:  # Standing
            if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):
                return False
            if (x, y) in self.obstacles:
                return False
                
        elif orientation == 1:  # Horizontal
            if not (0 <= x < self.grid_size[0] - 1 and 0 <= y < self.grid_size[1]):
                return False
            if (x, y) in self.obstacles or (x + 1, y) in self.obstacles:
                return False
            if (x, y) in self.weak_tiles or (x + 1, y) in self.weak_tiles:
                return False
                
        elif orientation == 2:  # Vertical
            if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] - 1):
                return False
            if (x, y) in self.obstacles or (x, y + 1) in self.obstacles:
                return False
            if (x, y) in self.weak_tiles or (x, y + 1) in self.weak_tiles:
                return False
        
        return True

    def _generate_obstacles(self) -> List[Tuple[int, int]]:
        """Generate obstacles that create a solvable path"""
        obstacles = []
        # Border
        for i in range(self.grid_size[0]):
            obstacles.append((i, 0))
            obstacles.append((i, self.grid_size[1] - 1))
        for j in range(self.grid_size[1]):
            obstacles.append((0, j))
            obstacles.append((self.grid_size[0] - 1, j))
        
        # Some internal obstacles that create a path
        if self.grid_size[0] > 6 and self.grid_size[1] > 6:
            # Create a maze-like structure
            for i in range(2, self.grid_size[0] - 2, 2):
                for j in range(2, self.grid_size[1] - 2, 2):
                    if random.random() < 0.3:  # 30% chance to add obstacle
                        obstacles.append((i, j))
        
        return list(set(obstacles))  # Remove duplicates