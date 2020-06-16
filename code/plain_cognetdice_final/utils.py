# Third party libraries
import numpy as np
import matplotlib.pyplot as plt

# My libraries
import graph_network


def draw_trained_model_line_response(policy, model_directory_path, number_of_nodes_per_agent, sim, title_suffix=""):
    for agent_index in range(sim.num_taggers):
        N_nodes = number_of_nodes_per_agent[agent_index]
        for q in range(N_nodes):
            observation_samples, next_nodes_likelihood = graph_network.show_node_input_response(joint_policy=policy, observation_interval=(-sim.max_size, sim.max_size), agent=agent_index, node=q, num_divisions=50)
            plt.figure()  # This is necessary to avoid different figures overlapping each other
            plt.plot(observation_samples, next_nodes_likelihood, marker='o')
            legend_labels = [f"Next node {next_q}" for next_q in range(N_nodes)]
            plt.legend(legend_labels)
            plt.xlabel("Observation")
            plt.ylabel("Next node likelihood")
            plt.ylim(0, 1.2)
            plt.title(f"Observation response for Agent {agent_index}, Node {q} ")
            plt.savefig(f"./{model_directory_path}/agent{agent_index}node{q}{title_suffix}lineplot.png")
            plt.close("all")


def draw_trained_model_bar_response(policy, model_directory_path, number_of_nodes_per_agent, sim, title_suffix=""):
    for agent_index in range(sim.num_taggers):
        N_nodes = number_of_nodes_per_agent[agent_index]
        for q in range(N_nodes):
            # Plot next node probabilities for the entire observation range:
            observation_samples, next_nodes_list = graph_network.show_node_input_response(joint_policy=policy, observation_interval=(-sim.max_size, sim.max_size), agent=agent_index, node=q, num_divisions=10)
            labels = [f"{obs:.1f}" for obs in observation_samples]
            print("Labels: ", labels)
            x = 2 * np.arange(len(labels))
            print("x: ", x)
            fig, ax = plt.subplots()

            bar_width = 0.2

            for next_q in range(N_nodes):
                ax.bar(x + next_q*bar_width, next_nodes_list[:, next_q], width=bar_width, label=f"Node node {next_q}")

            ax.set_xticks(x+N_nodes*bar_width/2)
            ax.set_xticklabels(labels)
            ax.legend()

            plt.xlabel("Observation")
            plt.ylabel("Next node likelihood")
            plt.ylim(0, 1.2)

            plt.title(f"Observation response for Agent {agent_index}, Node {q} ")
            plt.savefig(f"./{model_directory_path}/agent{agent_index}node{q}{title_suffix}barplot.png")


class Logger:
    def __init__(self, file_name):
        self.result_file = file_name

        with open(self.result_file, "w") as f:
            f.write("Instantiated the log file\n")

    def store(self, string_info):
        with open(self.result_file, "a") as f:
            f.write(string_info)
            f.write("\n")
