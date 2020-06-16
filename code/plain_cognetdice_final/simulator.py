import numpy as np
import copy


class State:
    def __init__(self, min_size, max_size, num_agents):
        self.agent_positions = np.random.uniform(min_size, max_size, num_agents)


class Simulator:
    def __init__(self, file_name):

        print("Instantiating a Simulator object using file " + file_name)
        with open(file_name, 'r') as simspecs_file:
            for file_line in simspecs_file:
                line_header = file_line.split(" ")[0].strip("\n")
                line_content = file_line.split(" ")[1:]
                if line_header == "minSize":
                    self.min_size = float(line_content[0])
                    print("minSize: ", self.min_size)
                elif line_header == "maxSize":
                    self.max_size = float(line_content[0])
                    print("maxSize: ", self.max_size)
                elif line_header == "numTaggers":
                    self.num_taggers = int(line_content[0])
                    self.num_agents = self.num_taggers + 1
                    print("numTaggers: ", self.num_taggers)
                    print("Number of agents: ", self.num_agents)
                elif line_header == "tagRange":
                    self.tag_range = [float(individual_range) for individual_range in line_content]
                    print("tagRange: ", self.tag_range)
                elif line_header == "obsError":
                    self.obs_error = [float(individual_obs_error) for individual_obs_error in line_content]
                    print("obsError: ", self.obs_error)
                elif line_header == "transitionError":
                    self.transition_error = [float(individual_trans_error) for individual_trans_error in line_content]
                    print("transitionError: ", self.transition_error)
                elif line_header == "agentSpeed":
                    self.agent_speed = [float(individual_speed) for individual_speed in line_content]
                    print("agentSpeed: ", self.agent_speed)
                elif line_header == "actions":
                    self.actions = []
                    for tagger_agent in range(self.num_taggers):
                        tagger_actions = simspecs_file.readline().strip("\n").split(" ")
                        self.actions.append(tagger_actions)
                    print("Actions: ", self.actions)
                elif line_header == "reward":
                    self.reward = float(line_content[0])
                    print("reward: ", self.reward)
                elif line_header == "penalty":
                    self.penalty = float(line_content[0])
                    print("penalty: ", self.penalty)
                elif line_header == "wait":
                    self.wait_penalty = float(line_content[0])
                    print("wait: ", self.wait_penalty)

    def num_actions(self, tagger_index):
        return len(self.actions[tagger_index])

    def get_action_name(self, agent_index, action_index):
        for index, action_name in enumerate(self.actions[agent_index]):
            if index == action_index:
                return action_name

    def perform_joint_action(self, state, actions_taken):
        dist = 0  # this variable is used to decide where the evader will go

        changed_state = copy.deepcopy(state)
        for tagger_index in range(self.num_taggers):
            agent_index = tagger_index + 1

            action_index = actions_taken[tagger_index]

            if (action_index == "tag" or action_index == 0) and abs(changed_state.agent_positions[agent_index] - changed_state.agent_positions[0]) <= self.tag_range[tagger_index]:  # successful tag
                pass
                #print("Agent %s tagged successfully!!!" % agent_index)

            elif action_index == "mvL1" or action_index == 1:  # move left
                changed_state.agent_positions[agent_index] -= self.agent_speed[agent_index]

            elif action_index == "mvR1" or action_index == 2:  # move right
                changed_state.agent_positions[agent_index] += self.agent_speed[agent_index]

            elif (action_index == "tag" or action_index == 0) and abs(changed_state.agent_positions[agent_index] - changed_state.agent_positions[0]) > self.tag_range[tagger_index]:  # unsuccessful tag
                pass
                #print("Failed tag! ")

            else:
                pass
                # print("Warning! Unrecognized action index %s " % action_index)

            # Introduce action error:
            changed_state.agent_positions[agent_index] = np.random.normal(changed_state.agent_positions[agent_index], self.transition_error[tagger_index])

            if changed_state.agent_positions[agent_index] > self.max_size:
                changed_state.agent_positions[agent_index] = self.max_size
            if changed_state.agent_positions[agent_index] < self.min_size:
                changed_state.agent_positions[agent_index] = self.min_size

            # Evader:
            dist += changed_state.agent_positions[agent_index] - changed_state.agent_positions[0]

        # Evader:
        if dist < 0:
            changed_state.agent_positions[0] += self.agent_speed[0]

        if dist > 0:
            changed_state.agent_positions[0] -= self.agent_speed[0]

        if changed_state.agent_positions[0] > self.max_size:
            changed_state.agent_positions[0] = self.max_size
        if changed_state.agent_positions[0] < self.min_size:
            changed_state.agent_positions[0] = self.min_size

        return changed_state

    def sample_initial_state(self):
        return State(self.min_size, self.max_size, self.num_agents)

    def sample_next_state(self, state, actions_taken):
        next_state = self.perform_joint_action(state, actions_taken)

        return next_state

    def sample_next_observation(self, next_state, actions_taken):
        observations_received = -1 * np.ones(self.num_taggers, dtype=float)

        for tagger in range(self.num_taggers):
            distance_to_evader = next_state.agent_positions[0] - next_state.agent_positions[tagger + 1]
            observations_received[tagger] = np.random.normal(distance_to_evader, self.obs_error[tagger])

        return observations_received

    def get_reward(self, state, actions_taken):
        joint_action_reward = 0
        for i in range(self.num_taggers):
            if actions_taken[i] == "tag" or actions_taken[i] == 0:  # index 0 means "tag"
                if abs(state.agent_positions[i + 1] - state.agent_positions[0]) <= self.tag_range[i]:
                    joint_action_reward += self.reward
                else:
                    joint_action_reward += self.penalty
            else:
                joint_action_reward += self.wait_penalty

        return joint_action_reward

