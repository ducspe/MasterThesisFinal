# Third party libraries
import time
import pickle
import os

# My libraries
from scene_simulator import ImageSimulator
import utils
import twod_cognetdice


if __name__ == "__main__":

    ############################################################
    # Define parameters:
    sim = ImageSimulator(nr_agents=2)
    number_of_nodes_per_agent = [4 for tagger in range(sim.num_agents)]
    number_of_actions_per_agent = [sim.num_actions(tagger) for tagger in range(sim.num_agents)]
    alpha = 0.2  # learning rate
    discount_factor = 1

    horizon = 4
    N_rollouts = 10
    N_iterations = 2
    N_top_rollouts = 5
    N_top_average = 2
    N_evals = 100  # how many cumulative rewards to use when calculating the average reward.
    N_epochs = 10
    batch_size = 2
    N_hidden_nodes = 160

    results_directory_name = f"results_data/experiment_h{horizon}_no{number_of_nodes_per_agent}_it{N_iterations}_rol{N_rollouts}_bero{N_top_rollouts}_beav{N_top_average}_ev{N_evals}_al{alpha}_df{discount_factor}_ep{N_epochs}_ba{batch_size}_hd{N_hidden_nodes}"

    try:
        os.mkdir(f"./{results_directory_name}")
    except OSError:
        pass
    ############################################################

    my_logger = utils.Logger(f"./{results_directory_name}/logs.txt")
    start_time = time.perf_counter()
    best_policy, best_value = twod_cognetdice.run_cognetdice(results_directory_name, my_logger, sim, number_of_nodes_per_agent, number_of_actions_per_agent, alpha, discount_factor, horizon, N_rollouts, N_iterations, N_top_rollouts, N_top_average, N_evals, N_epochs, batch_size, N_hidden_nodes)
    end_time = time.perf_counter()

    my_logger.store("Starting COGNetDICE at time %s" % start_time)
    my_logger.store("Ended COGNetDICE at time %s" % end_time)
    my_logger.store(f"The running COGNetDICE duration was {end_time-start_time} seconds")
    my_logger.store(f"The best COGNetDICE policy after all iterations is:\n{best_policy}")
    my_logger.store(f"The value of the best COGNetDICE policy is: {best_value}")

    with open(f"./{results_directory_name}/best_policy.obj", "wb") as f:
        pickle.dump(best_policy, f)



