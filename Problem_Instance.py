"""
.. module:: Problem_Instance
"""
import random


class ProblemInstance:
    def __init__(self, _number_of_agents, _number_of_distinct_items):
        self.number_of_agents = _number_of_agents
        self.number_of_distinct_items = _number_of_distinct_items
        self.item_list = None
        self.agent_valuations = None
        self.initial_allocation = None

    def is_valid_instance(self):
        if self.item_list is None or self.agent_valuations is None or self.initial_allocation is None:
            return False
        return True

    # Maybe use 'raise SomeError()' in stead of making assertions
    def set_item_list(self, item_list):
        assert (len(item_list) == self.number_of_distinct_items)
        self.item_list = item_list

    def set_agent_valuations(self, agent_valuations):
        assert (len(agent_valuations) == self.number_of_agents)
        assert (len(agent_valuations[0]) == self.number_of_distinct_items)
        self.agent_valuations = agent_valuations

    def set_bundle_matrix(self, bundle_matrix):
        assert (len(bundle_matrix) == self.number_of_agents)
        self.initial_allocation = bundle_matrix

    def compute_random_item_list(self, max_item_count):
        self.item_list = [random.randint(1, max_item_count) for x in range(self.number_of_distinct_items)]

    def compute_random_agent_valuations(self, max_valuation_for_an_item):
        self.agent_valuations = [[random.randint(1, max_valuation_for_an_item) for
                                  y in range(self.number_of_distinct_items)] for
                                 x in range(self.number_of_agents)]

    def compute_agents_valuations_for_item(self, item_index):
        return [self.agent_valuations[agent_index][item_index] for agent_index in range(self.number_of_agents)]

    # TODO: RENAME this, it returns the indices of the agents which have the highest valuations for an item
    def indices_of_agents_with_highest_valuations_for_item(self, item_index):
        agents_item_valuations = self.compute_agents_valuations_for_item(item_index)
        return [agent_index for agent_index in range(self.number_of_agents) if
                agents_item_valuations[agent_index] == max(agents_item_valuations)]

    def compute_suboptimal_Nash_welfare_allocation(self):
        def compute_agents_bundle_valuations():
            return [(self.agent_valuations[_agent_index][item_index] * _item_count
                     for _item_index, _item_count in bundle_allocation[_agent_index].items())
                    for _agent_index in range(self.number_of_agents)]

        # For each item; Find the agents who values the item the most. If there are multiple such agents then, allocate
        # the item in a way which minimizes the difference between the respective agents valuations of their own bundle.
        # Note: this implementation assumes that there isn't a single agent which
        #       values most of the items significantly higher than every other agent.
        bundle_allocation = [{} for agent in range(self.number_of_agents)]
        agents_valuations_of_the_bundles = [0 for agent in range(self.number_of_agents)]
        item_list = self.item_list.copy()
        for item_index in range(self.number_of_distinct_items):
            item_count_for_current_item = item_list[item_index]
            agent_valuations_for_current_item = self.compute_agents_valuations_for_item(item_index)
            greediest_agents = self.indices_of_agents_with_highest_valuations_for_item(item_index)
            while item_count_for_current_item > 0:
                # find an agent, among the group of who values the current item the highest, and who currently
                # finds their bundle to be the least valuable.
                greediest_agents_bundle_valuations = [
                    (agent_index, agents_valuations_of_the_bundles[agent_index])
                    for agent_index in greediest_agents
                ]
                highest_valued_item_lowest_valued_bundle = \
                    min(greediest_agents_bundle_valuations, key=lambda x: x[1])[0]

                # Add the item to their bundle
                if item_index not in bundle_allocation[highest_valued_item_lowest_valued_bundle]:
                    bundle_allocation[highest_valued_item_lowest_valued_bundle][item_index] = 1
                else:
                    bundle_allocation[highest_valued_item_lowest_valued_bundle][item_index] += 1

                agents_valuations_of_the_bundles[highest_valued_item_lowest_valued_bundle] += \
                    agent_valuations_for_current_item[highest_valued_item_lowest_valued_bundle]
                item_count_for_current_item -= 1
        self.initial_allocation = bundle_allocation

    def construct_problem_instance(self, max_amount_of_any_item, max_valuation_for_an_item):
        if not self.is_valid_instance():
            if self.item_list is None:
                self.compute_random_item_list(max_amount_of_any_item)
            if self.agent_valuations is None:
                self.compute_random_agent_valuations(max_valuation_for_an_item)
            if self.initial_allocation is None:
                self.compute_suboptimal_Nash_welfare_allocation()
