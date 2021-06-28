import copy

import efx.structures as efx_structure
import efx.allocator as efx_allocator
import numpy as np
import random
import logging
import time
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)
file = logging.FileHandler('log.csv')
logger.addHandler(file)
logging.basicConfig(level=logging.INFO)


def random_valuations(number_of_agents, total_number_of_items, max_valuation_for_an_item):
    valuations = efx_structure.Valuation(np.array([[random.randint(0, max_valuation_for_an_item)
                                                    for y in range(total_number_of_items)]
                                                   for x in range(number_of_agents)]))
    return valuations


def lists_to_allocation(bundle_allocation, number_of_agents):
    initial_allocation = efx_structure.Allocation(
        [efx_structure.Bundle(bundle_allocation[agent_index], agent_index)
         for agent_index in range(number_of_agents)]
    )
    return initial_allocation


def random_needing_problem(number_of_agents, total_number_of_items, max_valuation_for_an_item=10):
    valuations = random_valuations(number_of_agents, total_number_of_items, max_valuation_for_an_item)

    bundle_allocation = [[] for x in range(number_of_agents)]
    bundle_valuations = np.array([0 for agent in range(number_of_agents)])

    # Assign item to needing agent
    for item_index in range(total_number_of_items):
        valuations_of_current_item = valuations.agent_item_valuations[:, item_index]

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

    initial_allocation = lists_to_allocation(bundle_allocation, number_of_agents)

    return initial_allocation, random_valuations


def random_uneven_problem(number_of_agents, number_of_items, max_valuation_of_item=10):
    valuations = random_valuations(number_of_agents, number_of_items, max_valuation_of_item)
    bundle_allocation = [[] for x in range(number_of_agents)]

    for item in range(number_of_items):
        lucky_agent = random.randint(0, number_of_agents - 1)
        bundle_allocation[lucky_agent].append(item)

    initial_allocation = lists_to_allocation(bundle_allocation, number_of_agents)

    return initial_allocation, valuations


def random_even_steven(number_of_agents, number_of_items, max_valuation_of_item=10):
    valuations = random_valuations(number_of_agents, number_of_items, max_valuation_of_item)

    bundle_allocation = [[] for x in range(number_of_agents)]
    next_agent = 0
    for item in range(number_of_items):
        bundle_allocation[next_agent].append(item)
        next_agent = (next_agent + 1) % number_of_agents

    initial_allocation = lists_to_allocation(bundle_allocation, number_of_agents)

    return initial_allocation, valuations


def get_initial_number_of_matches(initial_allocation, valuations):
    n = initial_allocation.bundles.size
    g = efx_structure.EfxGraph(n, initial_allocation, valuations)
    m = efx_structure.Matching(n, initial_allocation, g.edges)
    return m.size


def measure_solve(delta, initial_allocation, valuations, number_of_agents, number_of_items):
    efx_alloc = efx_allocator.Allocator(delta, initial_allocation, valuations)

    # Measure values
    initial_nash_welfare = valuations.calculate_nash_welfare(initial_allocation)
    # initial_number_of_matches = get_initial_number_of_matches(initial_allocation, valuations)
    initial_number_of_matches = -1

    # Benchmark start
    time_start = time.time()
    efx_allocation, reallocation_counter = efx_alloc.find_efx_allocation()
    elapsed_time = time.time() - time_start
    # Benchmark end

    # Compute the result
    efx_nash_welfare = valuations.calculate_nash_welfare(efx_allocation)

    print("Agents:{}, Items:{}, Delta:{}".format(number_of_agents, number_of_items, delta))

    return (number_of_agents, number_of_items, initial_number_of_matches, initial_nash_welfare, efx_nash_welfare,
            elapsed_time, reallocation_counter, delta)


def run_single_test(delta, number_of_agents, number_of_items):
    # Setup instance
    # initial_allocation, valuations = random_needing_problem(number_of_agents, number_of_items)
    initial_allocation, valuations = random_uneven_problem(number_of_agents, number_of_items)
    # initial_allocation, valuations = random_even_steven(number_of_agents, number_of_items)

    result = measure_solve(delta, initial_allocation, valuations, number_of_agents, number_of_items)
    return result


def parameter_unwrapper(param):
    delta = param[0]
    number_of_agents = param[1]
    number_of_items = param[2]
    return run_single_test(delta, number_of_agents, number_of_items)


def test_params():
    number_of_iterations = 3
    delta = 0.05
    increment_agent_by = 10
    max_number_of_agents = 301

    params = []
    for number_of_agents in range(10, max_number_of_agents, increment_agent_by):
        item_amounts = [int(number_of_agents / 4),
                        int((number_of_agents / 4) * 3),
                        int(number_of_agents - 1),
                        int(number_of_agents),
                        int(number_of_agents * 2),
                        int(number_of_agents * 10),
                        int(number_of_agents * 30)]
        for number_of_items in item_amounts:
            for i in range(0, number_of_iterations):
                params.append((delta, number_of_agents, number_of_items))
    return params


def params_delta():  # list of delta values to try
    deltas = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    return deltas


def bulk_multiprocess_test():
    params = test_params()
    with ProcessPoolExecutor() as executor:
        results = executor.map(parameter_unwrapper, params)
    for r in results:
        info = "{:<8},{:<8},{:<8},{:<24},{:<24},{:<24},{:<4},{:<4}".format(r[0], r[1], r[2], r[3], r[4], r[5], r[6],
                                                                           r[7])
        logger.info(info)


def bulk_serial_test():
    params = test_params()
    for param in params:
        r = parameter_unwrapper(param)
        info = "{:<8},{:<8},{:<8},{:<24},{:<24},{:<24},{:<4},{:<4}".format(r[0], r[1], r[2], r[3], r[4], r[5], r[6],
                                                                           r[7])
        logger.info(info)


def bulk_same_problems():
    deltas = params_delta()
    number_of_problems = 10
    agent_item = [(5, 20), (5, 40), (5, 100), (30, 35), (30, 60), (30, 100), (30, 300), (100, 100), (100, 500),
                  (100, 2000), (300, 300), (300, 1000), (300, 3000)]

    problem_id = 0
    for pair in agent_item:
        for problem_number in range(0, number_of_problems):
            number_of_agents, number_of_items = pair
            initial_allocation, valuations = random_even_steven(number_of_agents, number_of_items)
            # initial_allocation, valuations = random_uneven_problem(number_of_agents, number_of_items)

            for delta in deltas:
                alloc = copy.deepcopy(initial_allocation)
                val = copy.deepcopy(valuations)
                r = measure_solve(delta, alloc, val, number_of_agents, number_of_items)
                info = "{:<8},{:<8},{:<8},{:<24},{:<24},{:<24},{:<4},{:<4}, {:<4}".format(r[0], r[1], r[2], r[3], r[4],
                                                                                          r[5], r[6],
                                                                                          r[7], problem_id)
                logger.info(info)
            problem_id += 1


def measure_solve_delta(delta, initial_allocation, valuations):
    efx_alloc = efx_allocator.Allocator(delta, initial_allocation, valuations)

    # Measure values
    initial_nash_welfare = valuations.calculate_nash_welfare(initial_allocation)
    efx_allocation, reallocation_counter, improved_input_alloc = efx_alloc.find_efx_allocation()

    # Compute the result
    efx_nash_welfare = valuations.calculate_nash_welfare(efx_allocation)
    realloc_welfare = valuations.calculate_nash_welfare(improved_input_alloc)

    return initial_nash_welfare, efx_nash_welfare, reallocation_counter, realloc_welfare


def bulk_delta_test():
    logger.info("\"problem_id\",\"delta\",\"initial_welfare\",\"reallocations\",\"realloc_welfare\",\"efx_welfare\"")

    deltas = params_delta()
    number_of_problems = 200
    agents = 50
    items = 300

    problem_id = 0
    for problem_number in range(0, number_of_problems):
        # initial_allocation, valuation = random_even_steven(agents, items)
        initial_allocation, valuation = random_uneven_problem(agents, items)

        for delta in deltas:
            alloc = copy.deepcopy(initial_allocation)
            val = copy.deepcopy(valuation)
            r = measure_solve_delta(delta, alloc, val)

            initial_welfare = r[0]
            reallocations = r[2]
            realloc_welfare = r[3]
            efx_welfare = r[1]

            logger.info("{}, {}, {}, {}, {}, {}".format(problem_id, delta, initial_welfare, reallocations,
                                                        realloc_welfare, efx_welfare))
        problem_id += 1


def main():
    """Again main
    :return:
    """
    # logger.info("\"Number of agents\", "
    #             + "\"Number of items\", "
    #             + "\"Number of initial matches\", "
    #             + "\"Initial Nash welfare\", "
    #             + "\"Efx Nash welfare\", "
    #             + "\"Computation time\", "
    #             + "\"Number of re-allocations\", "
    #             + "\"delta\", "
    #             + "\"Problem Id\"")

    # bulk_multiprocess_test()
    # bulk_serial_test()
    # bulk_same_problems()
    # single_even_test()
    bulk_delta_test()


if __name__ == '__main__':
    main()
