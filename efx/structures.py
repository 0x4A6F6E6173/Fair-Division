"""Efx helper classes.

This module contains helper classes for modeling different object types from the efx algorithm 2
of Caragiannis et.al. (2019).
The classes and functions in this module assume a valid input is
provided (from the allocator module). They perform no input validation themselves, and may lead
to errors if faulty input is provided.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List
from math import prod


class Bundle:
    """A bundle of some arbitrary number of items.

    Attributes:
        items (List[int]): The items currently in the bundle, modelled as a list of item ids.
        number (int): The bundle number, corresponding to the agent the bundle is initialized
            to. A bundle Z_j has number j.
        touched (bool): True if the bundle has been touched, false otherwise.

    Args:
        initial_items: The items the bundle contains at creation. List must contain ints.
        number: The bundle number.
    """
    def __init__(self, initial_items: List[int], number: int):
        self.items: List[int] = initial_items
        self.number = number
        self.touched = False

    def remove_item(self, item: int):
        """Remove a single item from bundle.

        Removes the specified item from the bundle. Also sets the touched attribute to true.

        Args:
            item: Id number of item to remove.
        """
        self.items.remove(item)
        self.touched = True

    @staticmethod
    def union(bundle_0: 'Bundle', bundle_1: 'Bundle', number: int) -> 'Bundle':
        """Join two bundles.
        
        Perform a union of bundles containing the union of the items of the two bundles. 
        Provides no guarantee on the order of items in the bundles. A new Bundle instance
        is returned, the two original bundles remain unchanged.

        Args:
            bundle_0: The first bundle to perform the union of.
            bundle_1: The second bundle to perform union of.
            number: The bundle number of the new bundle returned.

        Return:
            A new bundle instance, containing the union of the items of provided bundles.
        """
        items = list(set(bundle_0.items) | set(bundle_1.items))
        new_bundle = Bundle(items, number)
        return new_bundle

    @staticmethod
    def difference(bundle_0: 'Bundle', bundle_1: 'Bundle', number: int) -> 'Bundle':
        """Find the set difference of two bundles.
        
        Remove the items of bundle_1 from bundle 0. If an item is in bundle_1 but not
        bundle_0, it is ignored.
        Provides no guarantee on the order of items.
        A new bundle instance is returned, the parameter bundles remain unchanged.

        Args:
            bundle_0: The bundle to subtract from.
            bundle_1: The bundle to subtract.
            number: The bundle number of the new bundle.

        Returns:
            A new bundle instance, containing the difference of the two parameter bundles.
        """
        items = list(set(bundle_0.items) - set(bundle_1.items))
        new_bundle = Bundle(items, number)
        return new_bundle


class Allocation:
    """Representation of bundle allocation.

    Contains an array of bundles. Index j contains bundle B_j. The order should be
    the same as bundle numbers, so bundle B_j should have bundle.number = j.

    Attributes:
        bundles (List[Bundle]): The ordered bundles in the allocation.

    Args:
        bundles: A numpy array containing Bundle elements.
    """
    def __init__(self, bundles: List[Bundle]):
        self.bundles = np.array(bundles)


class Valuation:
    """Matrix-like representation of valuation function v.

    Represent the valuation function v as a 2d numpy array. If entry [i, j] = x,
    then agent i assigns a value of x to item j.

    This class also contains functions for working with bundles and allocations
    where item values factor into the operation. Requires calling initialize_bundle_values and
    initializa_cheap_lookup for some calculations, but does not check this is done.

    Attributes:
        agent_item_valuations (numpy.ndarray): 2d numpy array containing
            agent-item valuations.
        bundle_values (numpy.ndarray): stored bundle values to speed up calculations. Must be initialized
            and maintained.
        bundle_values_done (bool): True if using stored values, False otherwise.
        cheapest_for_bundles (numpy.ndarray): store the index of the cheapest item in a bundle, for all bundles,
            according to all agents. [i, bundle] is the index of the cheapest item in the list of items in the bundle
            according to agent i. To get the id of that item lookup items[[i, bundle]].
            To find cheapest item in bundle b according to agent i, lookup [i, b.number].
        cheapest_done (bool): True if using stored cheapest items, False otherwise.

    Args:
        valuations: 2d numpy array, agent on axis 0, item on axis 1.
    """

    def __init__(self, valuations: np.ndarray):
        self.agent_item_valuations = valuations

        self.bundle_values = None
        self.bundle_values_done = False

        self.cheapest_for_bundles = None
        self.cheapest_done = False

    def initialize_bundle_values(self, initial_allocation: Allocation):
        """Initialize stored bundle values to speed up calculations in other functions.

        Args:
            initial_allocation: all the bundles in of the problem.
        """
        number_of_agents = initial_allocation.bundles.size
        self.bundle_values = np.zeros((number_of_agents, number_of_agents))
        self.bundle_values_done = False

        for agent in range(number_of_agents):
            for bundle in initial_allocation.bundles:
                bundle_value = self.valuate_bundle(agent, bundle, force=True)
                self.bundle_values[agent, bundle.number] = bundle_value

        self.bundle_values_done = True

    def initialize_cheap_lookup(self, initial_allocation: Allocation):
        """Initialize lookup of cheapest item in bundles to speed up calculations in other functions.

        Args:
            initial_allocation: all the bundles of the problem.
        """
        number_of_agents = initial_allocation.bundles.size
        self.cheapest_for_bundles = np.full((number_of_agents, number_of_agents), -1)
        self.cheapest_done = False

        for agent in range(number_of_agents):
            for bundle in initial_allocation.bundles:
                index = self.index_least_valuable_item(agent, bundle, force=True)
                self.cheapest_for_bundles[agent, bundle.number] = index
        self.cheapest_done = True

    def valuate_bundle(self, agent: int, bundle: Bundle, force=False) -> float:
        """Calculate the value of bundle according to agent.

        If stored bundle values are done, just return a lookup. Otherwise do full calculation using items in
        bundle.

        Args:
            agent: Agent to valuate according to.
            bundle: Bundle to valuate.
            force: If True, do a complete recalculation. If False, lookup if possible. Defaults to False.

        Returns:
            The total value of the bundle. If bundle is empty, return value 0.
        """
        if self.bundle_values_done and (not force):
            return self.bundle_values[agent, bundle.number]
        else:
            if len(bundle.items) == 0:
                return 0
            values = self.agent_item_valuations[agent, bundle.items]
            return values.sum()

    def index_least_valuable_item(self, agent: int, bundle: Bundle, force=False) -> int:
        """Find the index in bundle item list of least valuable item according to agent.

        Args:
            agent: Agent to valuate according to.
            bundle: Bundle to find item in.
            force: Whether to do a full recalculation. If True do so, if False lookup stored. Defaults to False.

        Returns:
            The position index of the least valuable item.
        """
        index = -1

        if self.cheapest_done and (not force):
            index = self.cheapest_for_bundles[agent, bundle.number]
        elif len(bundle.items) > 0:
            item_values = self.agent_item_valuations[agent, bundle.items]
            index = np.argmin(item_values)
        return index

    def find_least_valuable_item(self, agent: int, bundle: Bundle) -> int:
        """Find the least valuable item in bundle according to agent.

        Args:
            agent: The agent to valuate according to.
            bundle: The bundle to find the item in.

        Returns:
            The id of the least valuable item.
        """
        index_of_least_valuable_item = self.index_least_valuable_item(agent, bundle)

        least_valuable_item = bundle.items[index_of_least_valuable_item]
        return least_valuable_item

    def valuate_bundle_item_removed(self, agent: int, bundle: Bundle) -> float:
        """ Valuate bundle when the least valuable item has been removed.

        Calculate the value of the bundle, and then subtract the value of the least valuable item.
        Original bundle is not changed.

        Args:
            agent: Agent to valuate according to.
            bundle: Bundle to remove from and valuate.

        Returns:
            Value of bundle after item has been removed. If bundle has no items, returns 0.
        """
        if len(bundle.items) == 0:
            return 0
        bundle_value = self.valuate_bundle(agent, bundle)
        least_valuable_item = self.find_least_valuable_item(agent, bundle)
        value_of_least_valuable_item = self.agent_item_valuations[agent, least_valuable_item]
        reduced_value = bundle_value - value_of_least_valuable_item

        return reduced_value

    def remove_min_val_item_from_bundle(self, agent: int, bundle: Bundle):
        """Update bundle by removing least valuable item.

        The least valuable item is found according to the specified agents valuations.
        Rather than returning a new bundle, this MODIFIES the contents of the specified bundle.
        If using stored bundle values, also updates those.
        If using stored cheapest, also update those.

        Args:
            agent: Agent to valuate according to.
            bundle: Bundle to modify.
        """
        # Find item to remove.
        least_valuable_item = self.find_least_valuable_item(agent, bundle)
        number_of_agents = np.shape(self.bundle_values)[0]
        bundle.remove_item(least_valuable_item)

        if self.bundle_values_done:  # Calculate and store values of bundle after item has been removed.
            for i in range(0, number_of_agents):
                old_bundle_value = self.bundle_values[i, bundle.number]
                value_of_item = self.agent_item_valuations[i, least_valuable_item]
                new_bundle_value = old_bundle_value - value_of_item
                self.bundle_values[i, bundle.number] = new_bundle_value

        if self.cheapest_done:  # Find the new cheapest item.
            for i in range(0, number_of_agents):
                index = self.index_least_valuable_item(i, bundle, force=True)
                self.cheapest_for_bundles[i, bundle.number] = index

    def is_bundle_efx_feasible_for_agent(self, agent: int, bundle: Bundle, allocation: Allocation) -> bool:
        """Determine if bundle is efx feasible for agent.

        Corresponds to condition (i) of efx feasibility graph.

        Args:
            agent: Agent to valuate according to and decide for.
            bundle: Bundle to evaluate.
            allocation: Current allocation of bundle. Used to access bundle contents.

        Returns:
            True if bundle is efx feasible for agent, False otherwise.
        """
        bundle_value = self.valuate_bundle(agent, bundle)
        for b in allocation.bundles:
            b_value = self.valuate_bundle_item_removed(agent, b)
            if b_value > bundle_value:
                return False
        return True

    def does_bundle_overrule_preferred(self, agent: int, bundle: Bundle, allocation: Allocation) -> bool:
        """Determine if bundle is strictly better than agents preferred.

        Determine truth of condition (ii) in definition of efx feasibility graph. If bundle is agents original,
        always return true. Otherwise compare bundle values.

        Args:
            agent: Agent to valuate according to.
            bundle: Bundle to evaluate.
            allocation: Current bundle allocation.

        Returns:
            True if bundle has a strictly better value, False otherwise. If bundle is the agents preferred,
            always return True.
        """
        is_own_bundle = agent == bundle.number
        if is_own_bundle:
            return True
        bundle_value = self.valuate_bundle(agent, bundle)
        own_bundle = allocation.bundles[agent]
        own_bundle_value = self.valuate_bundle(agent, own_bundle)
        if bundle_value > own_bundle_value:
            return True
        else:
            return False

    def robust_demand(self, agent: int, allocation: Allocation) -> Bundle:
        """Find the robust demand bundle of agent.

        Args:
            agent: Agent to find robust demand for.
            allocation: Current allocation of bundles.

        Returns:
            The robust demand bundle of the agent.
        """
        robust_demand = None
        robust_demand_value = -1
        for bundle in allocation.bundles:
            reduced_bundle_value = self.valuate_bundle_item_removed(agent, bundle)
            if reduced_bundle_value > robust_demand_value:
                robust_demand = bundle
                robust_demand_value = reduced_bundle_value
        return robust_demand

    def is_allocation_efx(self, allocation: Allocation) -> bool:
        """Check allocation is efx feasible for all agents.

        Args:
            allocation: Allocation of items.

        Returns:
            True if efx feasible, False otherwise.
        """
        for agent in range(0, allocation.bundles.size):
            bundle = allocation.bundles[agent]
            is_agent_satisfied = self.is_bundle_efx_feasible_for_agent(agent, bundle, allocation)
            if not is_agent_satisfied:
                return False
        return True

    def calculate_nash_welfare(self, allocation: Allocation) -> float:
        """Calculate the nash welfare of the allocation.

        Args:
            allocation: The allocation of items.

        Returns:
            The Nash welfare of the allocation under this valuation.
        """
        n: int = allocation.bundles.size
        valuaions = [np.power(self.valuate_bundle(i, allocation.bundles[i], force=True), (1 / n))
                        for i in range(n)]
        nash_welfare = prod(valuaions)
        return nash_welfare

    def calculate_log_nash_welfare(self, allocation: Allocation):
        """Calculates the log of an allocations nash welfare.
            This is primarily to combat the overflow of larger problem instances.
            But it also helps on the instances where a bundles nash welfare is 0

                Args:
                    allocation: The allocation of items.

                Returns:
                    The log Nash welfare of the allocation under this valuation.
                """

        def log(x):
            if x == 0:
                return 0
            else:
                return np.log10(x)

        n = allocation.bundles.size
        log_sum = sum(map(log, (self.valuate_bundle(xi, allocation.bundles[xi], force=True) for xi in range(0, n))))
        log_of_nash = (1 / n) * log_sum
        return log_of_nash


class Graph:
    """Graph helper class.

    Base class for graph-like structures. Contains a basic edge structure in the form of
    a 2d array. Agents along axis 0, bundles along axis 1. Is not updated when bundles change.
    Contains booleans values to signify connections. Edges are not directed.
    Used for efx feasibility graph and matchings.

    Attributes:
        edges (numpy.ndarray): The edges of the graph. If [i, j] is True agent i is connected to bundle j.
            If false there is no connection from i to j. 2d numpy array of booleans.
        agents (int): The total number of agents.

    Args:
        agents: Number of agents.
        edges: The edges of the graph. Must match the internal edges attribute.
    """

    def __init__(self, agents: int, edges: np.ndarray):
        self.edges = edges
        self.agents = agents

    def remove_edge(self, agent: int, bundle: int):
        """Remove an edge between agent and bundle.

        Args:
            agent: Agent to remove edge for.
            bundle: Bundle to remove edge for. Needs the int id of the bundle, not the bundle itself.
        """
        self.edges[agent, bundle] = False

    def add_edge(self, agent: int, bundle: int):
        """Add an edge between agent and bundle.

        Args:
            agent: Agent to add edge for.
            bundle: Bundle to add edge for. Needs the int id of the bundle, not the bundle itself.
        """
        self.edges[agent, bundle] = True


class EfxGraph(Graph):
    """Efx feasibility graph representation.

    Inherits basic edge functionality from Graph. This extension handles calculation of efx feasibility edges
    at instantiation in init. Agents along axis 0 and bundles along axis 1.
    Agent i is connected to bundle j if edges[i, j] = 1. Follows definition of E(G).
    To make use of claim 2, use update instead of constructing new graph.

    Args:
        number_of_agents: Total number of agent. Used also for number of bundles.
        allocation: How bundles are allocated to agents.
        valuation: How agents value items and therefore bundles.
    """

    def __init__(self, number_of_agents: int, allocation: Allocation, valuation: Valuation):
        number_of_bundles = number_of_agents
        edges = np.full((number_of_agents, number_of_bundles), False, dtype=bool)

        for agent in range(0, number_of_agents):
            for bundle in allocation.bundles:
                efx_feasible = valuation.is_bundle_efx_feasible_for_agent(agent, bundle, allocation)  # condition (i)
                overrule_preferred = valuation.does_bundle_overrule_preferred(agent, bundle, allocation)  # cond (ii)
                if efx_feasible and overrule_preferred:
                    edges[agent, bundle.number] = True

        Graph.__init__(self, number_of_agents, edges)

    def update(self, z_jstar: Bundle, allocation: Allocation, valuation: Valuation):
        """Update the Efx graph for next timestep.

        For any edge e=(i, Z_j) where j != jstar, use claim 2 to ignore edges already there from timestep t. Others are
        recalculated.

        Args:
            z_jstar: The bundle chosen as robust demand at time t.
            allocation: The allocation of bundles at time t+1.
            valuation: How agents values items and therefore bundles.
        """
        number_of_agents = self.agents
        for i in range(0, number_of_agents):
            for z_j in allocation.bundles:
                e_in_g = self.edges[i, z_j.number]
                z_j_is_z_jstar = z_j == z_jstar
                if e_in_g and (not z_j_is_z_jstar):
                    pass
                else:
                    efx_feasible = valuation.is_bundle_efx_feasible_for_agent(i, z_j,
                                                                              allocation)  # condition (i)
                    overrule_preferred = valuation.does_bundle_overrule_preferred(i, z_j,
                                                                                  allocation)  # cond (ii)
                    if efx_feasible and overrule_preferred:
                        self.edges[i, z_j.number] = True


class Matching(Graph):
    """Matching represented as a graph.

    If [i, j] = True, then agent i is matched to bundle j. Does not have to be a complete matching. Therefore at most
    a singe True for each column and row. Constructed using the idea of a weighted graph as suggested by Caragiannis
    et.al. (2019).
    Weights:
    Z_i is touched => incident edges w=n^4
    (i, Z_i) so bundle is original allocation => w = w + n^2
    else => w = 1

    Attributes:
        allocation (Allocation): How bundles are allocated to agents.
        size (int): The number of valid agent-bundle matches in the graph.

    Args:
        agents: Total number of agent. Used also for number of bundles.
        allocation: How bundles are allocated to agents.
        efx_edges: The edges from an efx feasibility graph, represented as a 2d array. If none, calculates the identity
        matching M0.
    """

    def __init__(self, agents: int, allocation: Allocation, efx_edges: np.ndarray = None):
        self.allocation = allocation
        edges = np.full((agents, agents), False, dtype=bool)
        self.size = 0
        if efx_edges is not None:  # Compute proper matching
            weights = self.weighted_graph(allocation, efx_edges)
            scipy_matching_rows = linear_sum_assignment(weights, maximize=True)[1]  # i == j means select edge [i, j]

            edges = np.full((agents, agents), False, dtype=bool)
            for agent in range(0, agents):
                bundle_match = scipy_matching_rows[agent]  # Bundle match might not be legal - edge must be in efx graph
                legal_match = efx_edges[agent, bundle_match]
                edges[agent, bundle_match] = legal_match
                self.size += legal_match

        else:  # Compute identity matching
            for agent in range(0, agents):
                edges[agent, agent] = True
            self.size = agents

        Graph.__init__(self, agents, edges)

    def get_unmatched_bundle(self) -> Bundle:
        """Retrieve an unmatched bundle.

        Returns:
            A bundle that is not matched to any agent. Chosen according to no particular heuristic.
        """
        is_bundles_matched = np.any(self.edges, axis=0)  # False if bundle i is unmatched
        unmatched_bundle_numbers = np.where(is_bundles_matched == False)[0]
        first_unmatched_bundle_number = unmatched_bundle_numbers[0]
        first_unmatched_bundle = self.allocation.bundles[first_unmatched_bundle_number]
        return first_unmatched_bundle

    def get_agent_matched_to_bundle(self, bundle: Bundle) -> int:
        """Get the agent matched to the bundle.

        Args:
            bundle: Bundle to find agent for. Must be matched, otherwise causes error.

        Returns:
            Agent id of agent.
        """
        bundle_edges = self.edges[:, bundle.number]
        agent = np.where(bundle_edges == True)[0][0]  # Matching, so only a single edge
        return agent

    def get_bundle_matched_to_agent(self, agent: int) -> Bundle:
        """Get bundle matched to agent.

        Args:
            agent: Agent to find bundle for. Must be matched, otherwise causes error.

        Returns:
            Matched bundle.
        """
        agent_edges = self.edges[agent, :]
        bundle_number = np.where(agent_edges == True)[0][0]  # Matching, so only a single edge
        bundle = self.allocation.bundles[bundle_number]
        return bundle

    def get_result_allocation(self, bundles_allocation: Allocation) -> Allocation:
        """Retrieve the allocation specified by the matching.

        Create a new allocation where bundles are assigned to agents according to the matching. Does not modify the
        existing allocation. Does not work for empty matchings.

        Args:
            bundles_allocation: Existing allocation, for getting hold of bundles.

        Returns:
            New allocation of bundles.
        """
        new_allocation_bundles = []
        for agent in range(0, self.size):
            agent_edges = self.edges[agent, :]
            bundle_number = np.where(agent_edges == True)[0][0]
            bundle = bundles_allocation.bundles[bundle_number]
            new_allocation_bundles.append(bundle)
        new_allocation = Allocation(new_allocation_bundles)
        return new_allocation

    def is_agent_matched(self, agent: int) -> bool:
        """Check if agent is matched.

        Args:
            agent: Agent to check for. Must be a valid agent for the current problem, otherwise causes errors.

        Returns:
            True if agent is matched, False otherwise.
        """
        agent_edges = self.edges[agent, :]
        is_matched = np.any(agent_edges)
        return is_matched

    @staticmethod
    def weighted_graph(allocation: Allocation, edges: np.ndarray) -> np.ndarray:
        """Construct a weighted version of the provided edge matrix.

        Agent on axis 0 and bundle number on axis 1. Touched bundles add weight n^4 to incident edges, original bundle
        edges add n^2 and remaining edges weight 1. Non-valid edges have weight 0.

        Args:
            allocation: Allocation of bundles. Needed to see if they are touched.
            edges: Boolean 2d numpy array, [i, j] == True means edge from agent i to bundle number j.

        Returns:
            2d numpy array of integer weights. Weighting depends on the number of agents.
        """
        agents = allocation.bundles.size
        number_of_bundles = agents
        weights = np.zeros((agents, number_of_bundles), dtype=int)
        for agent in range(0, agents):
            for bundle_number in range(0, number_of_bundles):
                does_edge_exist = edges[agent, bundle_number]
                if not does_edge_exist:
                    continue
                weights[agent, bundle_number] += 1

                is_bundle_touched = allocation.bundles[bundle_number].touched
                if is_bundle_touched:
                    weights[agent, bundle_number] += agents ** 4

                is_original_edge = agent == bundle_number
                if is_original_edge:
                    weights[agent, bundle_number] += agents ** 2
        return weights


class AugmentingPath:
    """Find and represent augmenting path in matching.

    Start from unmatched bundle Z_j1. Go along edge in M0.
    Then go along edge in M, then edge in M0, for as long as possible.
    End with an edge from M0, at unmatched agent jk.

     Path representation: 2d array of shape (length_of_path, 2)
        [[j1, Z_j2]
         [..., ...]
         [jk-1, Z_jk]]

    final_agent = jk

    Attributes:
        jk: The agent the final bundle in the path is allocated to, ie paired with in m0.
        path: The augmenting path of agents and bundles.

    Args:
        z_j1: The originating bundle of the path. Should not be matched.
        m: The actual computed matching for the current allocation.
        m0: The identity matching.
    """

    def __init__(self, z_j1: Bundle, m: Matching, m0: Matching):
        if m.size == m.agents:
            raise ValueError('Cannot compute augmenting path for a complete matching')

        p = []  # Don't know length before done, so use list for appending
        j1 = m0.get_agent_matched_to_bundle(z_j1)

        done = False
        working_agent = j1
        while not done:
            is_agent_matched = m.is_agent_matched(working_agent)  # follow edge in m
            if not is_agent_matched:
                break
            z_ji = m.get_bundle_matched_to_agent(working_agent)
            edge_i = (working_agent, z_ji)
            p.append(edge_i)

            working_agent = m0.get_agent_matched_to_bundle(z_ji)  # Follow edge in m0

        self.jk = working_agent
        p = np.array(p)  # Work with numpy arrays elsewhere. Path will no longer change length.
        self.path = p

    def is_bundle_in_path(self, bundle: Bundle) -> bool:
        """Check if bundle is in the path.

        Does not include the unmatched origin bundle z_j1.

        Args:
            bundle: Bundle to check for.

        Returns:
            True if bundle is in path, False otherwise. If path is empty, return False.
        """
        if self.path.size == 0:
            return False
        bundles_in_path = self.path[:, 1]
        checks = np.equal(bundles_in_path, bundle)
        in_path = np.any(checks)
        return in_path
