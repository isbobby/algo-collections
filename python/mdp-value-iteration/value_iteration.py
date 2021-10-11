import numpy as np

from maze import SAMPLE_MAZE


class MDP_Model():
    THRESHOLD = 0.05

    def __init__(self, maze) -> None:
        """
        Initialise using a simple map from maze.py. '.' denotes space, '*'
        denotes obstacle, 'G' denotes goal

        Attributes
            maze (np.matrix(float)) : 
                a matrix containing the expected reward for all cells in maze
            
            maze_height (int) 

            maze_length (int)
        """
        self.maze = self.initialise_maze(maze)
        self.policy = {}
        self.maze_height, self.maze_length = self.maze.shape
        pass

    def initialise_maze(self, maze):
        """
        Assign initial values to states, obstacles receive -1, goal receives +5,
        the other nodes are populated with 0
        """
        mdp_maze = []
        current_row = []
        for row in maze:
            for cell in row:
                if cell == '.':
                    current_row.append(0)
                elif cell == '*':
                    current_row.append(-1)
                elif cell == 'G':
                    current_row.append(5)
            mdp_maze.append(current_row)
            current_row = []

        mdp_maze = np.matrix(mdp_maze, dtype=float)
        return mdp_maze

    def value_iteration(self):
        """
        Run finite horizon value iteration, stops when the algorithm no longer 
        induce a change more than the set threshold
        """
        while True:
            biggest_change = 0
            for i in range(0, self.maze_height):
                for j in range(0, self.maze_length):
                    current_reward = self.maze[i, j]

                    if current_reward == -1 or current_reward == 5:
                        # Terminal states, do not iterate
                        continue

                    state = i, j
                    actions = self.get_actions(s=state)
                    new_reward = self.calculate_max_reward(
                        s=state, actions=actions)
                    self.maze[i, j] = max(self.maze[i, j], new_reward)
                    biggest_change = max(
                        biggest_change, new_reward - current_reward)

            if biggest_change < 0.05:
                break

        return

    def get_actions(self, s):
        """
        Return all valid actions the agent can take in a given cell, actions
        that bring the agent beyond the boundary of the map is removed
        """
        up = s[0] - 1, s[1]
        down = s[0] + 1, s[1]
        left = s[0], s[1] - 1
        right = s[0], s[1] + 1
        actions = [up, down, left, right]
        valid_actions = []
        for action in actions:
            if self.valid_cell(action):
                valid_actions.append(action)
        return valid_actions

    def valid_cell(self, s):
        """
        Check if the cell is within the confine of the maze
        """
        if s[0] < 0 or s[0] >= self.maze_height or s[1] < 0 or s[1] >= self.maze_length:
            return False
        return True

    def calculate_max_reward(self, s, actions):
        """
        Given a state s, and a range of actions a, calculate the maximum reward attainable
        from the given state and list of actions
        """
        max_reward = 0
        for action in actions:
            # record the action?
            max_reward = max(max_reward, self.reward(s, action))
        return max_reward

    def reward(self, s, a):
        """
        The reward function
        Given a state s, and action a, compute the expected reward

        Here, we assume the agent has 80% chance to get to the intended cell, and
        5% chance of going to every other cell (including being stationary)

        For some cells, the total probability does not sum up to 1 - this is because
        the agent is set not to move beyond the boundary of the maze
        """
        primary_action = a
        other_actions = self.get_actions(s)
        other_actions.remove(primary_action)

        # 80% of getting it right
        reward = 0.8 * self.maze[primary_action[0], primary_action[1]]

        # 5% of moving to unintended places
        for action in other_actions:
            reward += 0.05 * self.maze[action[0], action[1]]
        
        # 5% of being stationary
        reward += 0.05 * self.maze[s[0],s[1]]

        return reward

    def get_best_action(self):
        """
        Given a state s, retrieve the action from the policy
        """
        pass

    def visualize_maze(self):
        print(self.maze)


if __name__ == "__main__":
    model = MDP_Model(maze=SAMPLE_MAZE)
    model.value_iteration()
    model.visualize_maze()
