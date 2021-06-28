"""Top level module for finding efx allocations.

The Allocator class in this module is responsible for calling the helper functions from the structures module
to find an efx feasible allocation.
This also perform some basic, but not complete, validation of the input at class instantiation.
"""
import efx.structures as efx
import numpy as np
import copy


class Allocator:
    """Handler for finding an efx feasible solution.

    Attributes:
        delta (float): The delta value to use when checking if a new allocation x_hat is needed.
        initial_allocation (efx.Allocation): The original allocation provided at class instantiation.
            Can always be retrieved.
        allocation (efx.Allocation): The current working allocation. Is used by run_algorithm_once and updated
            by find_efx_allocation.
        valuations (efx.Valuation): The agent-item valuations used for computation. Does not change.

    Args:
        delta: Delta value, should be in range ??.
        initial_allocation: Initial allocation to work with.
        valuations: Agent-item valuations.
    """
    def __init__(self, delta: float, initial_allocation: efx.Allocation, valuations: efx.Valuation):
        Allocator.validate_input(delta, initial_allocation, valuations)
        self.delta = delta
        self.initial_allocation = initial_allocation
        self.allocation = initial_allocation
        self.valuations = valuations

    def run_algorithm_once(self) -> [efx.Allocation, bool]:
        """Run the efx "algorithm 2" once.

        Follows "Algorithm 2" by Caragiannis et.al. (2019).

        Uses the allocation attribute of the class instance as the input allocation X.
        Either finds an efx feasible allocation, or a new (non-efx) allocation with better Nash welfare.

        Returns:
            If efx solution found, return the efx feasible allocation and True. Otherwise return
            allocation with better Nash welfare and False.
        """
        initial_allocation = self.allocation
        self.valuations.initialize_bundle_values(initial_allocation)
        self.valuations.initialize_cheap_lookup(initial_allocation)

        # Setup, line 1 - 3
        n = initial_allocation.bundles.size
        x: efx.Allocation = copy.deepcopy(initial_allocation)
        z: efx.Allocation = copy.deepcopy(initial_allocation)
        delta1 = (2 * self.delta) / (1 - self.delta)

        m0 = efx.Matching(n, z, None)
        g = efx.EfxGraph(n, z, self.valuations)
        m = efx.Matching(n, z, g.edges)
        while not m.size == n:
            m = efx.Matching(n, z, g.edges)

            if m.size == n:
                break

            item_removed = False
            z_jstar = None

            while not item_removed:  # line 13
                z_j1 = m.get_unmatched_bundle()
                p = efx.AugmentingPath(z_j1, m, m0)
                jk = p.jk
                z_jstar = self.valuations.robust_demand(jk, z)
                z_jstar_in_p = p.is_bundle_in_path(z_jstar)
                if z_jstar_in_p:  # Matching M is changed, so assume it is sufficient to find match there
                    j0 = m.get_agent_matched_to_bundle(z_jstar)
                    m.remove_edge(j0, z_jstar.number)
                    m.add_edge(jk, z_jstar.number)
                else:
                    self.valuations.remove_min_val_item_from_bundle(jk, z_jstar)
                    jstar = z_jstar.number  # Should have identical index numbers

                    val_jstar_of_z_jstar = self.valuations.valuate_bundle(jstar, z_jstar, force=True)
                    val_jstar_of_x_jstar = self.valuations.valuate_bundle(jstar, x.bundles[jstar], force=True)

                    if (2 + delta1) * val_jstar_of_z_jstar < val_jstar_of_x_jstar and p.path.size > 0:  # Line 20
                        x_hat = self.compute_x_hat(x, z, p, z_jstar)

                        return x_hat, False  # Alg 2, line 26
                    item_removed = True
            g.update(z_jstar, z, self.valuations)
        y = m.get_result_allocation(z)
        return y, True  # Alg 2, line 32

    def find_efx_allocation(self) -> [efx.Allocation, int]:
        """Find efx allocation.

        Continuously find new allocations until an efx allocation is reached. The found solution is also stored in
        the allocation attribute.

        Returns:
            An efx allocation of the allocation attribute of the class instance, a count og how many times an
            allocation with improved Nash welfare, x hat, was calculated, and the last x hat computed
            (most improved input.) Returned as efx_allocation, reallocation_count, best_input.
        """
        done = False
        count_of_initial_allocations = 0
        while not done:
            allocation, done = self.run_algorithm_once()
            count_of_initial_allocations += 1
            if done:
                return allocation, count_of_initial_allocations, self.allocation

            self.allocation = allocation

    @staticmethod
    def compute_x_hat(x: efx.Allocation, z: efx.Allocation, p: efx.AugmentingPath, z_jstar: efx.Bundle) \
            -> efx.Allocation:
        """Move items between bundles to find X_hat. Does not modify existing bundles.

        Args:
            x: The original allocation X.
            z: The bundles after item has been removed.
            p: The augmenting path to base the item rearrangement on.
            z_jstar: The robust demand bundle of jk.

        Returns:
            A new allocation X_hat.
        """
        x_hat_bundles = copy.deepcopy(x.bundles)

        j1 = p.path[0, 0]
        x_j1 = x.bundles[j1]
        z_j2 = p.path[0, 1]
        x_hat1 = efx.Bundle.union(x_j1, z_j2, j1)
        x_hat_bundles[j1] = x_hat1  # Alg 2, line 21

        path_length = p.path.shape[0]
        k = path_length + 1
        for i in range(2, k - 1):
            ji = p.path[i, 0]
            x_ji = x.bundles[ji]
            z_ji = z.bundles[ji]
            z_ji_inc = p.path[i, 1]
            hat1 = efx.Bundle.difference(x_ji, z_ji, ji)
            hat2 = efx.Bundle.union(hat1, z_ji_inc, ji)
            x_hat_bundles[ji] = hat2  # Alg 2, line 22

        jk = p.jk
        x_jk = x.bundles[jk]
        z_jk = z.bundles[jk]
        jk_hat1 = efx.Bundle.difference(x_jk, z_jk, jk)
        jk_hat2 = efx.Bundle.union(jk_hat1, z_jstar, jk)
        x_hat_bundles[jk] = jk_hat2  # Alg 2, line 23

        jstar = z_jstar.number
        x_jstar = x.bundles[jstar]
        x_hat_jstar = efx.Bundle.difference(x_jstar, z_jstar, jstar)
        x_hat_bundles[jstar] = x_hat_jstar  # Alg 2, line 24

        for i in range(0, x.bundles.shape[0]):
            already_done_bundle = np.isin(i, p.path[:, 0]) or i == jk or i == jstar
            if not already_done_bundle:
                x_hat_bundles[i] = x.bundles[i]  # Alg 2, line 25

        x_hat = efx.Allocation(x_hat_bundles)
        return x_hat

    @staticmethod
    def validate_input(delta: float, initial_allocation: efx.Allocation, valuations: efx.Valuation):
        """Validate input to ensure efx calculation makes sense.

        Args:
            delta: Delta parameter for finding improved Nash welfare allocations.
            initial_allocation: The initial item allocation.
            valuations: Agent-item valuations.
        """
        # TODO: Verify delta is in required range.
        # TODO: Verify dimensions of input
        # TODO: All items in bundles are ints
        # TODO: No duplicate items
        if not isinstance(valuations.agent_item_valuations, np.ndarray):
            raise TypeError('numpy array expected, got {}'.format(type(valuations)))
        if not valuations.agent_item_valuations.ndim == 2:
            raise ValueError(
                'Valuations matrix must have dimension 2, but had {}'.format(valuations.agent_item_valuations.ndim))
        if not np.all(valuations.agent_item_valuations >= 0):
            raise ValueError('Valuations must be >= 0')
