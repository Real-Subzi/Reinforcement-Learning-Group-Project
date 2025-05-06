"""
This module implements a Continuous Random Walk environment and a Semi-gradient Sarsa agent
using tile coding for function approximation in reinforcement learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

class TileCoder:
    """
    A class that implements tile coding for function approximation in reinforcement learning.
    Tile coding is a method to convert continuous states into a discrete feature representation
    by using multiple overlapping tilings of the state space.
    """
    def __init__(self, num_tilings, tiles_per_tiling, low, high):
        """
        Initialize the tile coder with multiple overlapping tilings.
        
        Args:
            num_tilings (int): Number of overlapping tilings (helps with generalization)
            tiles_per_tiling (int): Number of tiles in each tiling
            low (float): Lower bound of the state space
            high (float): Upper bound of the state space
        """
        self.num_tilings = num_tilings
        self.tiles_per_tiling = tiles_per_tiling
        self.low = low
        self.high = high
        # Calculate the width of each tile
        self.tile_width = (high - low) / tiles_per_tiling
        # Create offsets for each tiling to ensure overlapping
        self.offsets = np.linspace(0, self.tile_width, num_tilings, endpoint=False)
    
    def get_tiles(self, state):
        """
        Convert a continuous state into active tile indices.
        
        Args:
            state (float): The current state value
            
        Returns:
            list: Active tile indices for each tiling
        """
        tiles = []
        # For each tiling, find which tile is activated by the current state
        for tiling in range(self.num_tilings):
            offset_state = state - self.offsets[tiling]
            tile_index = int((offset_state - self.low) / self.tile_width)
            # Ensure tile index is within bounds
            tile_index = max(0, min(tile_index, self.tiles_per_tiling - 1))
            # Calculate the actual tile index for this tiling
            tiles.append(tile_index + tiling * self.tiles_per_tiling)
        return tiles

class ContinuousRandomWalk:
    """
    An environment implementing a continuous random walk problem.
    The agent moves in a continuous state space with discrete actions.
    """
    def __init__(self):
        """Initialize the continuous random walk environment."""
        # Define state space bounds
        self.low = 0.0
        self.high = 1.0
        # Set initial state to middle of the space
        self.reset()
    
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            float: Initial state (0.5)
        """
        self.state = 0.5
        return self.state
    
    def step(self, action):
        """
        Take a step in the environment based on the chosen action.
        
        Args:
            action (int): 0 for left movement, 1 for right movement
            
        Returns:
            tuple: (next_state, reward, done)
        """
        # Movement size is random but direction depends on action
        movement = np.random.normal(loc=0.0, scale=0.1)
        if action == 0:  # Left
            movement = -abs(movement)
        else:  # Right
            movement = abs(movement)
        
        # Update state with movement
        self.state = self.state + movement
        self.state = max(self.low, min(self.high, self.state))
        
        # Check if episode is done (reached bounds)
        done = self.state <= self.low or self.state >= self.high
        
        # Calculate reward
        if done:
            reward = -1
        else:
            reward = 0
            
        return self.state, reward, done

class SarsaAgent:
    """
    An agent implementing Semi-gradient Sarsa algorithm with tile coding
    for function approximation.
    """
    def __init__(self, num_tilings=8, tiles_per_tiling=8):
        """
        Initialize the Sarsa agent with tile coding.
        
        Args:
            num_tilings (int): Number of tilings for tile coding
            tiles_per_tiling (int): Number of tiles per tiling
        """
        # Initialize tile coder for state space [0, 1]
        self.tile_coder = TileCoder(num_tilings, tiles_per_tiling, 0, 1)
        # Total number of features (tiles * 2 for two actions)
        self.num_features = num_tilings * tiles_per_tiling * 2
        # Initialize weights for function approximation
        self.weights = np.zeros(self.num_features)
        
    def get_features(self, state, action):
        """
        Get feature vector for a state-action pair using tile coding.
        
        Args:
            state (float): Current state
            action (int): Chosen action
            
        Returns:
            numpy.array: Feature vector (one-hot encoded tiles for the action)
        """
        # Get active tiles for the state
        active_tiles = self.tile_coder.get_tiles(state)
        # Initialize feature vector
        features = np.zeros(self.num_features)
        # Set features for active tiles based on action
        offset = len(active_tiles) * action
        for tile in active_tiles:
            features[tile + offset] = 2  # Multiply by 2 for better scaling
        return features
    
    def get_q(self, state, action):
        """
        Calculate Q-value for a state-action pair.
        
        Args:
            state (float): Current state
            action (int): Action to evaluate
            
        Returns:
            float: Estimated Q-value
        """
        # Q-value is dot product of features and weights
        return np.dot(self.get_features(state, action), self.weights)
    
    def get_action(self, state, epsilon):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state (float): Current state
            epsilon (float): Exploration rate
            
        Returns:
            int: Chosen action (0 or 1)
        """
        # Explore randomly with probability epsilon
        if np.random.random() < epsilon:
            return np.random.randint(2)
        # Otherwise choose the action with highest Q-value
        return np.argmax([self.get_q(state, 0), self.get_q(state, 1)])
    
    def update(self, state, action, reward, next_state, next_action, alpha):
        """
        Update weights using semi-gradient Sarsa update rule.
        
        Args:
            state (float): Current state
            action (int): Taken action
            reward (float): Received reward
            next_state (float): Next state
            next_action (int): Next action
            alpha (float): Learning rate
        """
        # Get feature vectors for current and next state-action pairs
        features = self.get_features(state, action)
        next_features = self.get_features(next_state, next_action)
        
        # Calculate TD error
        td_error = reward + np.dot(next_features, self.weights) - np.dot(features, self.weights)
        
        # Update weights using semi-gradient Sarsa update
        self.weights += alpha * td_error * features 