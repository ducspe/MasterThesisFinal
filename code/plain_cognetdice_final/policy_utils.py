# Third party libraries:
import numpy as np

# My libraries:
from graph_network import GNet

sim = "This simulation object should be set by the transfer_info function"
nr_hidden_nodes = "This represents the number of hidden nodes in the neural net and should be set by the transfer_info function"
results_directory_name = "This represents the name of the folder where various plots and logging data are stored about the ongoing experiment"
number_of_nodes_per_agent = "A list specifying what the user wants the number of nodes per agent to be"
my_logger = "Object that should log data to a persistent storage"


def transfer_info(simulation, N_hidden_nodes, results_directory, nr_nodes_list, logger):
    global sim, nr_hidden_nodes, results_directory_name, number_of_nodes_per_agent, my_logger
    sim = simulation
    nr_hidden_nodes = N_hidden_nodes
    results_directory_name = results_directory
    number_of_nodes_per_agent = nr_nodes_list
    my_logger = logger


def rollout(joint_policy, horizon, decpomdp_simulator, discount_factor):
    state = decpomdp_simulator.sample_initial_state()
    actions_taken = np.zeros(shape=(horizon, decpomdp_simulator.num_taggers), dtype=int)
    observations_received = np.zeros(shape=(horizon + 1, decpomdp_simulator.num_taggers), dtype=float)
    nodes_visited = np.zeros(shape=(horizon + 1, decpomdp_simulator.num_taggers), dtype=int)
    nodes_visited[0, :] = joint_policy.initial_nodes()
    cumulative_reward = 0.0
    for t in range(horizon):
        actions_taken[t, :] = joint_policy.sample_next_actions(nodes_visited[t, :])
        cumulative_reward += discount_factor**t * decpomdp_simulator.get_reward(state, actions_taken[t, :])
        state = decpomdp_simulator.sample_next_state(state, actions_taken[t, :])
        observations_received[t+1, :] = decpomdp_simulator.sample_next_observation(state, actions_taken[t, :])
        nodes_visited[t+1, :] = joint_policy.sample_next_nodes(nodes_visited[t, :], observations_received[t+1, :])

    return actions_taken, nodes_visited, observations_received, cumulative_reward


def evaluate_joint_policy(joint_policy, horizon, decpomdp_simulator, discount_factor, num_evaluations):
    rewards_list = []
    for eval_policy in range(num_evaluations):
        _, _, _, cumulative_reward = rollout(joint_policy, horizon, decpomdp_simulator, discount_factor)
        rewards_list.append(cumulative_reward)

    average_reward = np.average(rewards_list)
    return average_reward


def best_value_not_changed_for_the_past_k_iterations(best_policy_values_list, nr_past_iterations):
    print("best_policy_values_list[-k-1:-1]: ", best_policy_values_list[-nr_past_iterations-1:-1])
    if len(best_policy_values_list) >= nr_past_iterations:
        for item in best_policy_values_list[-nr_past_iterations-1:-1]:
            if item != best_policy_values_list[-1]:
                print("best_value_not_changed_for_the_past_k_iterations returns False")
                return False
    print("best_value_not_changed_for_the_past_k_iterations returns True")
    return True


def value_converged(joint_policy_values_history):
    recent_history_slice = joint_policy_values_history[-3:]

    if len(joint_policy_values_history) > 1 and np.std(recent_history_slice) < 0.5:
        print(f"std from value_converged: {np.std(recent_history_slice)}")
        my_logger.store(f"std from value_converged: {np.std(recent_history_slice)}")
        return True

    return False


def inject_entropy_without_collecting_new_data(joint_policy, decpomdp_simulator, theta_ei):
    print("-------------------------------Entropy Injection without collecting new data---------------------------------------------------")
    for tagger_agent in range(decpomdp_simulator.num_taggers):
        for current_node in range(joint_policy.local_policies[tagger_agent].num_nodes):
            maximum_entropy_distribution = np.ones_like(joint_policy.local_policies[tagger_agent].nodes[current_node].action_distribution) / float(joint_policy.local_policies[tagger_agent].num_actions)
            joint_policy.local_policies[tagger_agent].nodes[current_node].action_distribution = (1 - theta_ei) * joint_policy.local_policies[tagger_agent].nodes[current_node].action_distribution + theta_ei * maximum_entropy_distribution
            print("With entropy injection: Updated action distribution of node %s for agent %s is %s" % (current_node, tagger_agent, joint_policy.local_policies[tagger_agent].nodes[current_node].action_distribution))
            my_logger.store("With entropy injection: Updated action distribution of node %s for agent %s is %s" % (current_node, tagger_agent, joint_policy.local_policies[tagger_agent].nodes[current_node].action_distribution))


# A local policy node can either return an action distribution or the next node distribution for any observation
class LocalPolicyNode:
    def __init__(self, node_index, num_actions, num_nodes):
        self.action_distribution = np.ones(shape=num_actions, dtype=float) / float(num_actions)
        self.classifier = GNet(nr_hidden_nodes=nr_hidden_nodes, nr_output_nodes=num_nodes)

    def next_node_distribution(self, observation):
        return self.classifier.classify(observation)


# A policy for one agent
class LocalPolicy:
    def __init__(self, policy_index, num_nodes, num_actions):
        self.policy_index = policy_index
        self.num_actions = num_actions
        self.num_nodes = num_nodes
        self.nodes = [LocalPolicyNode(node_index, num_actions, num_nodes) for node_index in range(num_nodes)]

    def initial_node(self):
        return 0

    def sample_next_action(self, current_node):
        action_distribution = self.nodes[current_node].action_distribution
        return np.random.choice(self.num_actions, p=action_distribution)

    def sample_next_node(self, current_node, local_observation):
        node_distribution = self.nodes[current_node].next_node_distribution(local_observation)
        return np.random.choice(self.num_nodes, p=node_distribution)

    def __str__(self):
        line = "##################Local Policy##################\n"
        for node_index, policy_node in enumerate(self.nodes):
            line += f"Node {node_index} prefers action {sim.get_action_name(agent_index=self.policy_index, action_index=np.argmax(policy_node.action_distribution))}\n"
            #line += f"Node {node_index} has action distribution: {policy_node.action_distribution}\n"

        line += "################End Local Policy################\n"
        return line


# A joint policy collects the policies of many agents and provides a convenient interface
class JointPolicy:
    def __init__(self, num_nodes_per_agent, num_actions_per_agent):
        self.local_policies = [LocalPolicy(index, x, y) for index, (x, y) in enumerate(zip(num_nodes_per_agent, num_actions_per_agent))]
        self.num_local_policies = len(self.local_policies)

    def initial_nodes(self):
        nodes = np.zeros(shape=self.num_local_policies, dtype=int)
        for index, local_policy in enumerate(self.local_policies):
            nodes[index] = local_policy.initial_node()
        return nodes

    def sample_next_actions(self, current_nodes):
        actions = np.zeros(shape=self.num_local_policies, dtype=int)
        for index, (local_policy, q) in enumerate(zip(self.local_policies, current_nodes)):
            actions[index] = local_policy.sample_next_action(q)
        return actions

    def sample_next_nodes(self, current_nodes, observations):
        next_nodes = np.zeros(shape=self.num_local_policies, dtype=int)
        for index, (local_policy, q_current, obs) in enumerate(zip(self.local_policies, current_nodes, observations)):
            next_nodes[index] = local_policy.sample_next_node(q_current, obs)
        return next_nodes

    def __str__(self):
        line = "-------------------Joint Policy------------------\n"
        for agent_index, local_policy in enumerate(self.local_policies):
            line += f"Agent {agent_index} has policy:\n{local_policy}\n"
        line += "-----------------End Joint Policy----------------\n"
        return line

    def set_simulation_deterministic_attributes(self, action_indices, current_node_indices, horizon):
        flag_node = np.full((self.local_policies[0].num_nodes, self.num_local_policies), False)

        for tagger_agent in range(self.num_local_policies):
            for iterated_node in range(self.local_policies[tagger_agent].num_nodes):
                for t in range(horizon):
                    if current_node_indices[t, tagger_agent] == iterated_node:
                        if not flag_node[iterated_node][tagger_agent]:
                            self.local_policies[tagger_agent].nodes[current_node_indices[t, tagger_agent]].action_distribution = np.zeros(shape=(self.local_policies[tagger_agent].num_actions))

                        np.put(self.local_policies[tagger_agent].nodes[current_node_indices[t, tagger_agent]].action_distribution, action_indices[t, tagger_agent], 1)

                        if flag_node[iterated_node][tagger_agent]:
                            self.local_policies[tagger_agent].nodes[current_node_indices[t, tagger_agent]].action_distribution /= np.sum(self.local_policies[tagger_agent].nodes[current_node_indices[t, tagger_agent]].action_distribution)

                        flag_node[iterated_node][tagger_agent] = True
