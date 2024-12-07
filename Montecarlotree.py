import numpy as np
import gymnasium as gym
from typing import List, Optional, Tuple

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children: List[Node] = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = self.get_possible_actions()

    def is_terminal(self) -> bool:
        # Check if state is terminal (game over)
        return self.state.terminated

    def get_reward(self) -> float:
        # Return reward from environment
        return self.state.reward

class MCTS:
    def __init__(self, env: gym.Env, exploration_weight=1.0):
        self.env = env
        self.exploration_weight = exploration_weight
        self.root = None

    def select_action(self, state, time_limit: float) -> any:
        """Main MCTS method to select the best action from current state."""
        self.root = Node(state)
        
        # Run iterations until time limit
        while time_remaining():  # You'll need to implement time checking
            leaf = self._select()
            simulation_result = self._simulate(leaf)
            self._backpropagate(leaf, simulation_result)
            
        # Return best action based on visit counts
        return self._best_child(self.root).state

    def _select(self) -> Node:
        """Selection step: traverse tree using UCB until reaching a leaf node."""
        node = self.root
        while node.untried_actions == [] and node.children != []:
            node = self._ucb_select(node)
        
        # Expansion
        if node.untried_actions != []:
            action = np.random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            new_state = self._get_next_state(node.state, action)
            child = Node(new_state, parent=node)
            node.children.append(child)
            return child
            
        return node

    def _simulate(self, node: Node) -> float:
        """Simulation step: use rollout policy to simulate until terminal state."""
        current_state = node.state
        while not self._is_terminal(current_state):
            action = self._rollout_policy(current_state)
            current_state = self._get_next_state(current_state, action)
        return self._get_reward(current_state)

    def _backpropagate(self, node: Node, reward: float):
        """Backup step: update values from leaf to root."""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _ucb_select(self, node: Node) -> Node:
        """Select child using UCB1 formula."""
        log_n_visits = np.log(node.visits)
        
        def ucb_value(child: Node) -> float:
            exploitation = child.value / child.visits
            exploration = self.exploration_weight * np.sqrt(log_n_visits / child.visits)
            return exploitation + exploration

        return max(node.children, key=ucb_value)

    def _rollout_policy(self, state) -> any:
        """Random action selection for rollout."""
        return self.env.action_space.sample()

    def _best_child(self, node: Node) -> Node:
        """Select best child based on visit count."""
        return max(node.children, key=lambda c: c.visits)

    def _get_next_state(self, state, action):
        """Step environment with action."""
        return self.env.step(action)

    def _get_possible_actions(self, state) -> List:
        """Get valid actions from environment."""
        return list(range(self.env.action_space.n))

    def _is_terminal(self, state) -> bool:
        """Check if state is terminal."""
        return state.terminated

    def _get_reward(self, state) -> float:
        """Get reward from state."""
        return state.reward
