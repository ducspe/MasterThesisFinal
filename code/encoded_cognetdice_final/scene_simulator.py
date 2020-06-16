import numpy as np
import copy

# Dimensions of the environment:
SCENE_WIDTH = 16
SCENE_HEIGHT = 16
PATCH_WIDTH = 4
PATCH_HEIGHT = 4


class State:
    def __init__(self, num_agents):
        self.agent_positions = [[np.random.randint(0, SCENE_HEIGHT-PATCH_HEIGHT//2) - 1, np.random.randint(0, SCENE_WIDTH-PATCH_WIDTH//2) - 1] for agent in range(num_agents)]


class ImageSimulator:

    # This function is used to generate a Gaussian that will subsequently be transformed into a heatmap
    @staticmethod
    def generate_scene(x, y, peak_offset_x, peak_offset_y, add_noise=False, noise_mu=0, noise_std=0.005, noise_amplitude=1):
        sx = 5
        sy = 5
        noise = noise_amplitude * np.random.normal(noise_mu, noise_std, size=1)
        func = np.exp(-((x - peak_offset_x) ** 2 / (2 * sx ** 2) + (y - peak_offset_y) ** 2 / (2 * sy ** 2)))

        if add_noise:
            return noise + func
        else:
            return func

    def __init__(self, nr_agents):
        self.num_agents = nr_agents
        self.scene = np.zeros(shape=(SCENE_HEIGHT, SCENE_WIDTH))

        self.scene_height = SCENE_HEIGHT
        self.scene_width = SCENE_WIDTH
        self.patch_height = PATCH_HEIGHT
        self.patch_width = PATCH_WIDTH

        self.source_position = [0, 0]  # Position of the static source that emits the signal.

        for x in range(self.scene_height):
            for y in range(self.scene_width):
                pixel_value = self.generate_scene(x, y, self.source_position[0], self.source_position[1])
                self.scene[x][y] = pixel_value

    def num_actions(self, agent):
        return 4  # move up, down, left, right

    def get_action_name(self, agent_index, action_index):
        if action_index == 0:
            return "mR"
        elif action_index == 1:
            return "mL"
        elif action_index == 2:
            return "mU"
        elif action_index == 3:
            return "mD"
        else:
            return "Unknown action"

    def perform_joint_action(self, state, actions_taken, agent_speed):

        changed_state = copy.deepcopy(state)

        for agent in range(self.num_agents):
            action = actions_taken[agent]

            if action == 0:  # move right
                changed_state.agent_positions[agent][0] += agent_speed
            elif action == 1:  # move left
                changed_state.agent_positions[agent][0] -= agent_speed
            elif action == 2:  # move up
                changed_state.agent_positions[agent][1] -= agent_speed
            elif action == 3:  # move down
                changed_state.agent_positions[agent][1] += agent_speed
            else:
                print("Warning! Action index unknown!!!!!")

            # If the agent gets outside of the environment boundaries, reset its position to be at the boundary
            if changed_state.agent_positions[agent][0] >= self.scene_height - self.patch_height//2:
                changed_state.agent_positions[agent][0] = self.scene_height - self.patch_height//2 - 1

            if changed_state.agent_positions[agent][0] < self.patch_height//2:
                changed_state.agent_positions[agent][0] = self.patch_height//2

            if changed_state.agent_positions[agent][1] >= self.scene_width - self.patch_width//2:
                changed_state.agent_positions[agent][1] = self.scene_width - self.patch_width//2 - 1

            if changed_state.agent_positions[agent][1] < self.patch_width//2:
                changed_state.agent_positions[agent][1] = self.patch_width//2

        return changed_state

    def sample_initial_state(self):
        initial_state = State(self.num_agents)
        return initial_state

    def sample_next_state(self, state, actions_taken):
        next_state = self.perform_joint_action(state, actions_taken, agent_speed=1)
        return next_state

    def sample_next_observation(self, state, actions_taken):
        observations_received = np.zeros(shape=(self.num_agents, self.patch_height, self.patch_width))
        vision_range = np.zeros(shape=(self.patch_height, self.patch_width))
        for agent in range(self.num_agents):
            x = state.agent_positions[agent][0]
            y = state.agent_positions[agent][1]
            vision_range[:, :] = self.scene[x-self.patch_height//2:x+self.patch_height//2, y-self.patch_width//2:y+self.patch_width//2]
            observations_received[agent, :, :] = vision_range

        return observations_received

    def get_reward(self, state, actions_taken):
        joint_action_reward = 0
        immediate_vicinity_radius = 3
        close_vicinity_radius = 4
        for agent in range(self.num_agents):
            if (state.agent_positions[agent][0] - self.source_position[0])**2 + (state.agent_positions[agent][1] - self.source_position[1])**2 < immediate_vicinity_radius**2:
                joint_action_reward += 50
            elif (state.agent_positions[agent][0] - self.source_position[0])**2 + (state.agent_positions[agent][1] - self.source_position[1])**2 < close_vicinity_radius**2:
                joint_action_reward += 1
            else:
                joint_action_reward -= 1

        return joint_action_reward