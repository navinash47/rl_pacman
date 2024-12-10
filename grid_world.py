import numpy as np

class GridWorld:
    def __init__(self):
        # Grid dimensions
        self.rows = 5
        self.cols = 5
        self.states = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        # Define states
        self.obstacles = [(2, 2), (3, 2)]  # Obstacle states
        self.water = (4, 2)                # Water state
        self.goal = (4, 4)                 # Goal state
        
        # Define actions
        self.actions = ['AU', 'AD', 'AL', 'AR']  # Up, Down, Left, Right
        
        # Action probabilities
        self.p_intended = 0.8    # Probability of moving in intended direction
        self.p_right = 0.05
        self.p_left = 0.05
        self.p_stay = 0.10      # Probability of staying in place
        
        # Current state
        self.current_state = (0, 0)  # Start at State 1
        
    def is_valid_state(self, state):
        """Check if state is valid (within bounds and not obstacle)"""
        r, c = state
        return (0 <= r < self.rows and 
                0 <= c < self.cols and 
                state not in self.obstacles)
    
    def get_next_state(self, state, action):
        """Get next state based on current state and action"""
        r, c = state
        
        # Define movement directions (up, down, left, right)
        movements = {
            'AU': (-1, 0),
            'AD': (1, 0),
            'AL': (0, -1),
            'AR': (0, 1)
        }
        
        # Get intended movement
        dr, dc = movements[action]
        new_r, new_c = r + dr, c + dc
        
        # Check if move is valid
        if self.is_valid_state((new_r, new_c)):
            return (new_r, new_c)
        return state  # Stay in current state if invalid move
    
    def step(self, action):
        """Take a step in the environment"""
        if action not in self.actions:
            raise ValueError("Invalid action")
            
        # Determine actual movement based on probabilities
        p = np.random.random()
        
        if p < self.p_stay:  # 10% chance to stay
            actual_action = None
        elif p < self.p_stay + self.p_intended:  # 80% chance for intended action
            actual_action = action
        elif p < self.p_stay + self.p_intended + self.p_right:
            if action == 'AU':
                actual_action = 'AR'
            elif action == 'AD':
                actual_action = 'AL'
            else:
                actual_action = 'AU'
        else:
            if action == 'AU':
                actual_action = 'AL'
            elif action == 'AD':
                actual_action = 'AR'
            else:
                actual_action = 'AD'
        
        # Update state
        if actual_action:
            self.current_state = self.get_next_state(self.current_state, actual_action)
        
        # Calculate reward
        if self.current_state == self.goal:
            reward = 10
        elif self.current_state == self.water:
            reward = -10
        else:
            reward = 0
        
        # Check if episode is done
        done = self.current_state == self.goal
        
        return self.current_state, reward, done
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_state = (0, 0)
        return self.current_state

    def get_q_value_for_vi(self, state, action, course, gamma, V):
        """Get the q value for the given state and action"""
        if course == "intended":
            next_state, reward, _ = self.step(action)
            return reward + gamma * V[next_state]
        elif course == "right":
            next_state, reward, _ = self.step(self.right_of(action))
            return reward + gamma * V[next_state]
        elif course == "left":
            next_state, reward, _ = self.step(self.left_of(action))
            return reward + gamma * V[next_state]
        elif course == "stay":
            # Calculate reward for staying in the same state
            if state == self.goal:
                reward = 10
            elif state == self.water:
                reward = -10
            else:
                reward = 0
            return reward + gamma * V[state]
    def right_of(self,action):
        """Get the right of the given state"""
        if action == 'AU':
            return 'AR'
        elif action == 'AD':
            return 'AL'
        elif action == 'AL':
            return 'AU'
        elif action == 'AR':
            return 'AD'
    def left_of(self,action):
        """Get the left of the given state"""
        if action == 'AU':
            return 'AL'
        elif action == 'AD':
            return 'AR'
        elif action == 'AL':
            return 'AD'
        elif action == 'AR':
            return 'AU'

    


