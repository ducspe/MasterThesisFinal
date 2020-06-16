# Third party libraries
import numpy as np
import sys
import copy

# My libraries
import twod_graph_network
import twod_policy_utils


def run_cognetdice(results_directory_name, my_logger, sim, number_of_nodes_per_agent, number_of_actions_per_agent, alpha, discount_factor, horizon, N_rollouts, N_iterations, N_top_rollouts, N_top_average, N_evals, N_epochs, batch_size, N_hidden_nodes):
    v_b = -sys.float_info.max  # stores the value of the best policy
    best_policy_values_list = []
    joint_policy_values_history = []  # Keep track of the joint policy values to see if the value converged in the past k iterations
    best_policy = None

    INJECT_ENTROPY_FLAG = False
    theta_ei = 0.03  # parameter for entropy injection

    twod_policy_utils.transfer_info(sim, N_hidden_nodes, results_directory_name, number_of_nodes_per_agent, my_logger)

    joint_policy = twod_policy_utils.JointPolicy(number_of_nodes_per_agent, number_of_actions_per_agent)

    ####################################################################################################################
    # Store the initial weights of the neural nets for each agent and node:
    for agent in range(sim.num_agents):
        for current_node in range(joint_policy.local_policies[agent].num_nodes):
            model_file_path = f"./{results_directory_name}/a{agent}n{current_node}_initialweights"
            twod_graph_network.save_node_model(
                network_model=joint_policy.local_policies[agent].nodes[current_node].classifier,
                filepath=model_file_path)

    ####################################################################################################################

    actions_taken = np.zeros(shape=(N_rollouts, horizon, sim.num_agents), dtype=int)
    nodes_visited = np.zeros(shape=(N_rollouts, horizon + 1, sim.num_agents), dtype=int)
    observations_received = np.zeros(shape=(N_rollouts, horizon + 1, sim.num_agents, sim.patch_height, sim.patch_width), dtype=float)
    cumulative_reward = np.zeros(shape=N_rollouts, dtype=float)

    for iteration_k in range(N_iterations):

        for sim_index in range(N_rollouts):
            actions_taken[sim_index, :, :], nodes_visited[sim_index, :, :], observations_received[sim_index, :, :, :, :], cumulative_reward[sim_index] = twod_policy_utils.rollout(joint_policy, horizon, sim, discount_factor)

        top_sim_indices = np.argsort(cumulative_reward)[-N_top_rollouts:]
        print("Cumulative rewards: ", cumulative_reward)
        print("Top cumulative rewards: ", cumulative_reward[top_sim_indices])

        best_sim_index_list = []
        for top_sim_index in top_sim_indices:
            candidate_joint_policy_for_improvement = copy.deepcopy(joint_policy)
            candidate_joint_policy_for_improvement.set_simulation_deterministic_attributes(actions_taken[top_sim_index, :, :], nodes_visited[top_sim_index, :, :], horizon)
            candidate_value = twod_policy_utils.evaluate_joint_policy(candidate_joint_policy_for_improvement, horizon, sim, discount_factor, N_evals)

            if candidate_value > v_b:
                best_sim_index_list.append((top_sim_index, candidate_value))
                print(f"New candidate value for improvement {candidate_value}")

        best_average_experience_list = sorted(best_sim_index_list, key=lambda x: x[1])[-N_top_average:]

        if len(best_average_experience_list) == 0:
            twod_policy_utils.inject_entropy_without_collecting_new_data(joint_policy, sim, theta_ei)
            print(f"Continuing to next iteration {iteration_k+1} because the best experience list is empty.")
            print(f"Current best value is {v_b}")
            print(f"Value of the joint policy at the moment is {joint_policy_values_history[-1]}")
            continue

        print("Best experience list: ", best_average_experience_list)

        # Top simulations:
        best_simulation_indices = [item[0] for item in best_average_experience_list]
        final_best_indices = np.array(best_simulation_indices, dtype=int).reshape(len(best_simulation_indices))

        avg_best_actions = actions_taken[final_best_indices]
        avg_best_nodes = nodes_visited[final_best_indices]
        avg_best_observations = observations_received[final_best_indices]

        for agent in range(sim.num_agents):
            for current_node in range(joint_policy.local_policies[agent].num_nodes):
                q_visited = avg_best_nodes[:, :-1, agent] == current_node
                q_visited_next = np.concatenate((np.full((len(best_average_experience_list), 1), False, dtype=bool), q_visited), axis=1)

                counts = np.bincount(avg_best_actions[q_visited, agent], minlength=joint_policy.local_policies[agent].num_actions)
                n = counts.sum()

                if n != 0:
                    updated_action_distrib = counts.astype(float) / n

                    if INJECT_ENTROPY_FLAG:
                        maximum_entropy_distribution = np.ones_like(joint_policy.local_policies[agent].nodes[current_node].action_distribution) / float(joint_policy.local_policies[agent].num_actions)
                        joint_policy.local_policies[agent].nodes[current_node].action_distribution = (1-theta_ei) * (alpha * updated_action_distrib + (1 - alpha) * joint_policy.local_policies[agent].nodes[current_node].action_distribution) + theta_ei * maximum_entropy_distribution
                        print("With entropy injection: Updated action distribution of node %s for agent %s is %s" % (current_node, agent, joint_policy.local_policies[agent].nodes[current_node].action_distribution))
                        my_logger.store("With entropy injection: Updated action distribution of node %s for agent %s is %s" % (current_node, agent, joint_policy.local_policies[agent].nodes[current_node].action_distribution))

                    else:
                        joint_policy.local_policies[agent].nodes[current_node].action_distribution = alpha * updated_action_distrib + (1 - alpha) * joint_policy.local_policies[agent].nodes[current_node].action_distribution
                        print("No entropy injection: Updated action distribution of node %s for agent %s is %s" % (current_node, agent, joint_policy.local_policies[agent].nodes[current_node].action_distribution))
                        my_logger.store("No entropy injection: Updated action distribution of node %s for agent %s is %s" % (current_node, agent, joint_policy.local_policies[agent].nodes[current_node].action_distribution))

                x_train = avg_best_observations[q_visited_next, agent]
                y_train = avg_best_nodes[q_visited_next, agent]

                if len(x_train) > 0:
                    twod_graph_network.train(x_input=x_train, y_output=y_train, num_epochs=N_epochs, batch_size=batch_size, network_model=joint_policy.local_policies[agent].nodes[current_node].classifier)

        if INJECT_ENTROPY_FLAG:
            INJECT_ENTROPY_FLAG = False

        # Evaluate the policy:
        joint_policy_value = twod_policy_utils.evaluate_joint_policy(joint_policy, horizon, sim, discount_factor, N_evals)
        best_policy_values_list.append(v_b)

        joint_policy_values_history.append(joint_policy_value)
        if twod_policy_utils.value_converged(joint_policy_values_history):
            INJECT_ENTROPY_FLAG = True

        if twod_policy_utils.best_value_not_changed_for_the_past_k_iterations(best_policy_values_list, 3):
            INJECT_ENTROPY_FLAG = True

        # Update best value and store best_policy
        if joint_policy_value > v_b:
            v_b = joint_policy_value
            best_policy = copy.deepcopy(joint_policy)
            print(f"Found a new best policy with value {v_b}:\n{best_policy} at iteration {iteration_k}")
            my_logger.store(f"Found a new best policy with value {v_b}:\n{best_policy} at iteration {iteration_k}")

        print(f"End of iteration {iteration_k}")
        print(f"The joint policy after iteration {iteration_k} has value:\n{joint_policy_value}")
        print(f"The best value overall after iteration {iteration_k} is {v_b}")
        my_logger.store(f"End of iteration {iteration_k}")
        my_logger.store(f"The joint policy after iteration {iteration_k} has value:\n{joint_policy_value}")
        my_logger.store(f"The best value overall after iteration {iteration_k} is {v_b}")

    # End of iteration k

    # Write the weights of each neural net:
    for agent in range(sim.num_agents):
        for current_node in range(joint_policy.local_policies[agent].num_nodes):
            model_file_path = f"./{results_directory_name}/a{agent}n{current_node}_finalweights"
            twod_graph_network.save_node_model(
                network_model=joint_policy.local_policies[agent].nodes[current_node].classifier,
                filepath=model_file_path)

    return best_policy, v_b
