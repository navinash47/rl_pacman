import numpy as np

class catVsMonsters:
    def __init__(self):
        # Grid dimensions
        self.rows = 5
        self.cols = 5
        self.states = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        
        # Define states
        self.furniture = [(2, 1), (2, 2), (2, 3), (3, 2)]  # Obstacles
        self.monsters = [(4, 1), (0, 3)]  # Danger state
        self.food = [(4, 4)]  # Goal state
        self.goal = (4, 4)  # Explicit goal state
        
        # Define actions
        self.actions = ['AU', 'AD', 'AL', 'AR']  # Match GridWorld actions
        self.action_effects = {
            'AU': (-1, 0),
            'AD': (1, 0),
            'AL': (0, -1), 
            'AR': (0, 1)
        }
        
        # Action probabilities
        self.p_intended = 0.7   # 70% intended direction
        self.p_right = 0.12     # 12% for right deviation
        self.p_left = 0.12      # 12% for left deviation
        self.p_stay = 0.06      # 6% stay in place
        
        # Initialize current state
        self.current_state = (0, 0)
        
    def get_next_state(self, state, action):
        """Get next state based on current state and action"""
        if action not in self.actions:
            raise ValueError("Invalid action")
            
        # Get movement vector
        dr, dc = self.action_effects[action]
        new_r = state[0] + dr
        new_c = state[1] + dc
        
        # Check if move is valid
        if (new_r >= 0 and new_r < self.rows and 
            new_c >= 0 and new_c < self.cols and
            (new_r, new_c) not in self.furniture):
            return (new_r, new_c)
        return state
    
    def step(self, action):
        """Take a step in the environment"""
        if action not in self.actions:
            raise ValueError("Invalid action")
            
        # Determine actual movement based on probabilities
        p = np.random.random()
        
        if p < self.p_stay:  # 6% chance to stay
            actual_action = None
        elif p < self.p_intended + self.p_stay:  # 70% chance for intended action
            actual_action = action
        elif p < self.p_intended + self.p_stay + self.p_left:  # 12% chance for left deviation
            if action == 'AU':
                actual_action = 'AL'
            elif action == 'AD':
                actual_action = 'AR'
            else:
                actual_action = 'AU'
        else:  # 12% chance for right deviation
            if action == 'AU':
                actual_action = 'AR'
            elif action == 'AD':
                actual_action = 'AL'
            else:
                actual_action = 'AD'
        
        # Update state
        if actual_action:
            self.current_state = self.get_next_state(self.current_state, actual_action)
        
        # Calculate reward
        if self.current_state == self.goal:
            reward = 10
        elif self.current_state in self.monsters:
            reward = -8
        else:
            reward = -0.05  # Small step penalty
        
        # Check if episode is done
        done = self.current_state == self.goal
        
        return self.current_state, reward, done
            
    def reset(self):
        """Reset environment to initial state"""
        self.current_state = (0, 0)
        return self.current_state

    def get_q_value_for_vi(self, state, action, course="intended", gamma=0.9, V=None):
        """Get the q value for the given state and action"""
        if course == "intended":
            # Calculate reward based on next state
            next_state = self.get_next_state(state, action)
            reward = self._get_reward(next_state)
            return reward + gamma * V[next_state]
        elif course == "right":
            next_state = self.get_next_state(state, self.right_of(action))
            reward = self._get_reward(next_state)
            return reward + gamma * V[next_state]
        elif course == "left":
            next_state = self.get_next_state(state, self.left_of(action))
            reward = self._get_reward(next_state)
            return reward + gamma * V[next_state]
        elif course == "stay":
            # Calculate reward for staying in the same state
            reward = self._get_reward(state)
            return reward + gamma * V[state]

    def _get_reward(self, state):
        """Helper method to calculate reward for a given state"""
        if state == self.goal:
            return 10
        elif state in self.monsters:
            return -8
        else:
            return -0.05

    def right_of(self, action):
        """Get the right of the given action"""
        if action == 'AU':
            return 'AR'
        elif action == 'AD':
            return 'AL'
        elif action == 'AL':
            return 'AU'
        elif action == 'AR':
            return 'AD'

    def left_of(self, action):
        """Get the left of the given action"""
        if action == 'AU':
            return 'AL'
        elif action == 'AD':
            return 'AR'
        elif action == 'AL':
            return 'AD'
        elif action == 'AR':
            return 'AU'

    
