import efx.structures as efx_structure
import efx.allocator as efx_allocator
import concurrent.futures
import random
import numpy as np
import time



def compute_random_needing_problem(number_of_agents, total_number_of_items, max_valuation_for_an_item=10):
    random_valuations = efx_structure.Valuation(np.array([[random.randint(0, max_valuation_for_an_item)
                                                           for y in range(total_number_of_items)]
                                                          for x in range(number_of_agents)]))

    bundle_allocation = [[] for x in range(number_of_agents)]
    bundle_valuations = np.array([0 for agent in range(number_of_agents)])

    # Assign item to needing agent
    for item_index in range(total_number_of_items):
        valuations_of_current_item = random_valuations.agent_item_valuations[:, item_index]

        max_item_valuation = max(valuations_of_current_item)
        agents_with_highest_item_valuations = np.where(valuations_of_current_item == max_item_valuation)
        lowest_bundle_valuation = min(bundle_valuations[agents_with_highest_item_valuations])

        poorest_agent = -1
        for agent_index in range(number_of_agents):
            item_is_valued_most = valuations_of_current_item[agent_index] == max_item_valuation
            bundle_is_valued_least = bundle_valuations[agent_index] == lowest_bundle_valuation
            if item_is_valued_most and bundle_is_valued_least:
                poorest_agent = agent_index
                break

        bundle_allocation[poorest_agent].append(item_index)
        bundle_valuations[poorest_agent] += valuations_of_current_item[poorest_agent]

    initial_allocation = efx_structure.Allocation(
        [efx_structure.Bundle(bundle_allocation[agent_index], agent_index)
         for agent_index in range(number_of_agents)]
    )

    return initial_allocation, random_valuations


def compute_random_uneven_problem(number_of_agents, number_of_items, max_valuation_of_item = 10):
    random_valuations = efx_structure.Valuation(np.array([[random.randint(0, max_valuation_of_item)
                                                           for y in range(number_of_items)]
                                                          for x in range(number_of_agents)]))

    bundle_allocation = [[] for x in range(number_of_agents)]

    for item in range(number_of_items):
        lucky_agent = random.randint(0, number_of_agents-1)
        bundle_allocation[lucky_agent].append(item)

    initial_allocation = efx_structure.Allocation(
        [efx_structure.Bundle(bundle_allocation[agent_index], agent_index)
         for agent_index in range(number_of_agents)]
    )

    return initial_allocation, random_valuations


def run_test_power(delta, factor_of_agents, factor_of_items, step_size):
    # Setup instance
    number_of_agents = np.power(step_size, factor_of_agents)
    number_of_items = np.power(step_size, factor_of_items)
    initial_allocation, valuations = compute_random_uneven_problem(number_of_agents, number_of_items)
    efx_alloc = efx_allocator.Allocator(delta, initial_allocation, valuations)

    # Measure values
    initial_log_nash_welfare = valuations.calculate_nash_welfare(initial_allocation)

    # Benchmark start
    # possible options: time.perf_counter, time.process_time, time.thread_time, and time.clock
    time_start = time.thread_time()
    efx_allocation, reallocation_counter = efx_alloc.find_efx_allocation()
    elapsed_time = time.thread_time() - time_start
    # Benchmark end

    # Compute the result
    efx_log_nash_welfare = valuations.calculate_nash_welfare(efx_allocation)

    # Give the user a proof-of-work
    print("Agents:{:<7}, Items:{:<7}, Delta:{}".format(number_of_agents, number_of_items, delta))
    return number_of_agents, number_of_items, delta, initial_log_nash_welfare, efx_log_nash_welfare, elapsed_time, reallocation_counter


def get_initial_number_of_matches(initial_allocation, valuations):
    n = initial_allocation.bundles.size
    g = efx_structure.EfxGraph(n, initial_allocation, valuations)
    m = efx_structure.Matching(n, initial_allocation, g.edges)
    return m.size


def write_result_to_file(log_file, num_agents, num_items, num_match, initial_nash, efx_nash, elapsed, counter, delta):
    log_file.write("{:<8},{:<8},{:<8},{:<24},{:<24},{:<24},{:<4},{:<4}\n".format(
        num_agents,
        num_items,
        num_match,
        initial_nash,  # Initial Nash welfare
        efx_nash,  # EFX Nash welfare
        elapsed,  # Elapsed time
        counter,  # reallocation_counter
        delta
    ))


def run_test_threaded(file, delta, number_of_agents, number_of_items):
    # Setup instance
    initial_allocation, valuations = compute_random_needing_problem(number_of_agents, number_of_items)
    efx_alloc = efx_allocator.Allocator(delta, initial_allocation, valuations)

    # Measure values
    initial_nash_welfare = valuations.calculate_nash_welfare(initial_allocation)
    initial_number_of_matches = get_initial_number_of_matches(initial_allocation, valuations)

    # Benchmark start
    # possible options: time.perf_counter, time.process_time, time.thread_time, and time.clock
    time_start = time.time()
    efx_allocation, reallocation_counter = efx_alloc.find_efx_allocation()
    elapsed_time = time.time() - time_start
    # Benchmark end

    # Compute the result
    efx_nash_welfare = valuations.calculate_nash_welfare(efx_allocation)

    # Give the user a proof-of-work
    print("Agents:{}, Items:{}, Delta:{}".format(number_of_agents, number_of_items, delta))

    write_result_to_file(file, number_of_agents, number_of_items, initial_number_of_matches, initial_nash_welfare,
                         efx_nash_welfare, elapsed_time, reallocation_counter, delta)


def run_test(file, delta, number_of_agents, number_of_items):
    # Setup instance
    initial_allocation, valuations = compute_random_needing_problem(number_of_agents, number_of_items)
    efx_alloc = efx_allocator.Allocator(delta, initial_allocation, valuations)

    # Measure values
    initial_nash_welfare = valuations.calculate_nash_welfare(initial_allocation)
    initial_number_of_matches = get_initial_number_of_matches(initial_allocation, valuations)

    # Benchmark start
    # possible options: time.perf_counter, time.process_time, time.thread_time, and time.clock
    time_start = time.time()
    efx_allocation, reallocation_counter = efx_alloc.find_efx_allocation()
    elapsed_time = time.time() - time_start
    # Benchmark end

    # Compute the result
    efx_nash_welfare = valuations.calculate_nash_welfare(efx_allocation)

    # Give the user a proof-of-work
    print("Agents:{}, Items:{}, Delta:{}".format(number_of_agents, number_of_items, delta))

    write_result_to_file(file, number_of_agents, number_of_items, initial_number_of_matches, initial_nash_welfare,
                         efx_nash_welfare, elapsed_time, reallocation_counter, delta)


def test_iteration_set(number_of_iterations: int, delta, number_of_agents, number_of_items, file):
    for number_of_iterations in range(number_of_iterations):
        run_test(file, delta, number_of_agents, number_of_items)


def test_agent_item_relation_threaded():
    number_of_iterations = 10
    delta = 0.1

    log_file = open("./log_file.txt", "w")
    log_file.write("Number of agents, "
                    + "Number of items, "
                    + "Number of initial matches, "
                    + "Initial Nash welfare, "
                    + "Efx Nash welfare, "
                    + "Computation time, "
                    + "Number of re-allocations, "
                    + "delta\n")

    increment_agent_by = 20
    max_number_of_agents = 220
    for number_of_agents in range(100, max_number_of_agents, increment_agent_by):
        # Try with quarter items
        number_of_items = int(number_of_agents / 4)
        test_iteration_set(number_of_iterations, delta, number_of_agents, number_of_items, log_file)

        # Try with half items
        number_of_items = int(number_of_agents / 2)
        test_iteration_set(number_of_iterations, delta, number_of_agents, number_of_items, log_file)

        # Try with 3 quarters items
        number_of_items = int((number_of_agents / 4) * 3)
        test_iteration_set(number_of_iterations, delta, number_of_agents, number_of_items, log_file)

        # Try with 1 less item
        number_of_items = int(number_of_agents - 1)
        test_iteration_set(number_of_iterations, delta, number_of_agents, number_of_items, log_file)

        # Try with same number of items
        number_of_items = int(number_of_agents)
        test_iteration_set(number_of_iterations, delta, number_of_agents, number_of_items, log_file)

        # Try with 1 more item
        number_of_items = int(number_of_agents + 1)
        test_iteration_set(number_of_iterations, delta, number_of_agents, number_of_items, log_file)

        # Try with double items
        number_of_items = int(number_of_agents * 2)
        test_iteration_set(number_of_iterations, delta, number_of_agents, number_of_items, log_file)

        # Try with 10 times items
        number_of_items = int(number_of_agents * 10)
        test_iteration_set(number_of_iterations, delta, number_of_agents, number_of_items, log_file)

        # Try with 30 times items
        number_of_items = int(number_of_agents * 30)
        test_iteration_set(number_of_iterations, delta, number_of_agents, number_of_items, log_file)

    log_file.close()


def run_test_in_serial(factor_of_agents, factor_of_items, step_size=10):
    number_of_iterations = 10
    # run over delta size?
    log_file = open("./log_file_serial.txt", "w")
    log_file.write("Nash welfare of the initial allocation, "
                   + "Nash welfare of the computed EFX allocation, "
                   + "Time the EFX allocation took to compute, "
                   + "How many iterations the computation went through, "
                   + "delta, and "
                   + "iteration number."
                   + "Every {}'th line contains the average\n".format(number_of_iterations))
    for number_of_agents in range(1, factor_of_agents):
        for number_of_items in range(number_of_agents, factor_of_items):
            for delta_factor in range(100):
                delta = 1 - (0.01 * delta_factor)
                print("Computing instance with: {}#Agents, {}#items, {}#Delta...".format(
                    np.power(step_size, number_of_agents),
                    np.power(step_size, number_of_items),
                    delta
                ))
                sum_of_initial_nash_welfare = 0
                sum_of_elapsed_time = 0
                sum_of_efx_nash_welfare = 0
                sum_of_recomputation_counts = 0

                for curr_iteration in range(number_of_iterations):
                    data = run_test_power(delta, number_of_agents, number_of_items, step_size)
                    # Increase the sum totals
                    sum_of_initial_nash_welfare += data[0]
                    sum_of_efx_nash_welfare += data[1]
                    sum_of_elapsed_time += data[2]
                    sum_of_recomputation_counts += data[3]

                    # Write current iteration to file
                    log_file.write("{:<4},{:<24},{:<24},{:<24},{:<4},{:<4}\n".format(
                        curr_iteration,
                        data[0],  # Initial Nash welfare
                        data[1],  # EFX Nash welfare
                        data[2],  # Elapsed time
                        delta,  # Delta
                        data[3]  # reallocation_counter
                    ))
                    print("  - Thread#{} Finished after {:<4} iterations at {}".format(curr_iteration,
                                                                                       data[3],
                                                                                       data[2]))
                # Write approx. averages to file
                log_file.write("{:<24},{:<24},{:<24},{:<4}\n".format(
                    sum_of_initial_nash_welfare / number_of_iterations,
                    sum_of_efx_nash_welfare / number_of_iterations,
                    sum_of_elapsed_time / number_of_iterations,
                    sum_of_recomputation_counts / number_of_iterations,
                ))
                print("           Finished at an average of {}".format(sum_of_elapsed_time / number_of_iterations))
    log_file.close()


def run_test_in_parallel(factor_of_agents, factor_of_items, step_size=10):
    number_of_threads = 8
    number_of_iterations = 3
    log_file = open("./log_file_parallel.txt", "w")
    log_file.write("Number of agents, "
                   + "Number of items, "
                   + "delta, "
                   + "Nash welfare of the initial allocation, "
                   + "Nash welfare of the computed EFX allocation, "
                   + "Time the EFX allocation took to compute, and "
                   + "Number of calls to the algorithm.\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_threads) as executor:
        future_results = []
        for number_of_agents in range(1, factor_of_agents + 1):
            for number_of_items in range(1, factor_of_items + 1):
                for delta_factor in range(0, 100, 10):
                    for iterations in range(number_of_iterations):
                        delta = 1 - (0.01 * delta_factor)
                        future_results.append(executor.submit(run_test_power, delta,
                                                              number_of_agents, number_of_items,
                                                              step_size))

        for future_result in concurrent.futures.as_completed(future_results):
            try:
                data = future_result.result()
            except Exception as exc:
                print("A thread generated an exception {}".format(exc))
            else:
                # Write current iteration to file
                log_file.write("{:<8},{:<8},{:<22},{:<24},{:<24},{:<24},{:<4}\n".format(
                    data[0],  # Number of agents
                    data[1],  # Number of items
                    data[2],  # Delta
                    data[3],  # Initial Nash welfare
                    data[4],  # EFX Nash welfare
                    data[5],  # Elapsed time
                    data[6]  # Reallocation_counter
                ))

    log_file.close()


def main():
    """Again main
    :return:
    """
    # test for agents = {10, 100, 1000}, unique_items = {10, 100, 1000}, number_of_items = {10, 100, 100}
    # run_test_in_parallel(7, 7)
    # run_test_in_serial(6, 6)
    test_agent_item_relation_threaded()


if __name__ == '__main__':
    main()
