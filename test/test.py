"""Unittests for project.

This module contains unittests produced during the implementation of the project.
Does not currently test the functions where the result is not predetermined - like the function finding a
efx allocation in the efx.Allocator class, which may choose untouched bundles in any order, and therefore reach
different results.
"""
import unittest
import numpy as np
import efx.structures as structures
import efx.allocator as allocator
from unittest.mock import Mock
from typing import List


def same_contents(arr1: List[int], arr2: List[int]) -> bool:
    """Check if arr1 and arr2 have the same contents regardless of order, so fx [0, 1, 2] and [2, 1, 0] yield true."""
    sort1 = np.sort(arr1)
    sort2 = np.sort(arr2)
    result = np.array_equal(sort1, sort2)
    return result


class TestBundle(unittest.TestCase):
    def test_remove_item(self):
        bundle = structures.Bundle([0, 1, 2, 3], 0)

        bundle.remove_item(2)

        expected_items = [0, 1, 3]
        result = same_contents(expected_items, bundle.items)
        self.assertTrue(result)

    def test_union(self):
        bundle_0 = structures.Bundle([0, 1, 2, 3], 0)
        bundle_1 = structures.Bundle([0, 1, 4, 5], 1)

        union = structures.Bundle.union(bundle_0, bundle_1, 0)

        expected = [0, 1, 2, 3, 4, 5]
        result = same_contents(expected, union.items)
        self.assertTrue(result)

    def test_difference_1(self):
        bundle_0 = structures.Bundle([0, 1, 2, 3], 0)
        bundle_1 = structures.Bundle([2, 0], 1)

        difference = structures.Bundle.difference(bundle_0, bundle_1, 0)

        expected_items = [1, 3]
        result = same_contents(expected_items, difference.items)
        self.assertTrue(result)

    def test_difference_2(self):
        bundle_0 = structures.Bundle([0, 1, 2, 3, 4], 0)
        bundle_1 = structures.Bundle([0, 1, 2, 3], 0)

        difference = structures.Bundle.difference(bundle_0, bundle_1, 0)

        expected_items = np.array([4])
        self.assertTrue(np.array_equal(difference.items, expected_items))


class TestAllocation(unittest.TestCase):
    def test_init(self):
        bundle_0 = structures.Bundle([0, 1, 2], 0)
        bundle_1 = structures.Bundle([5, 4], 1)
        bundle_2 = structures.Bundle([8, 7], 2)
        bundles = [bundle_0, bundle_1, bundle_2]

        allocation = structures.Allocation(bundles)

        self.assertEqual(allocation.bundles[0], bundle_0)
        self.assertEqual(allocation.bundles[1], bundle_1)
        self.assertEqual(allocation.bundles[2], bundle_2)


class TestValuationStored(unittest.TestCase):
    def setUp(self):
        self.item_values = np.array([[0, 0, 0, 0, 0], [10, 7, 5, 5, 5], [1, 2, 3, 4, 5]])
        self.agent = 1
        self.bundle = structures.Bundle([2, 0, 4], 0)
        self.bundle_rest_1 = structures.Bundle([1, 3], 1)
        self.bundle_rest_2 = structures.Bundle([], 2)
        self.allocation = structures.Allocation([self.bundle, self.bundle_rest_1, self.bundle_rest_2])
        self.valuation = structures.Valuation(self.item_values)
        self.valuation.initialize_bundle_values(self.allocation)
        self.valuation.initialize_cheap_lookup(self.allocation)

    def test_init_cheap(self):
        expected = np.array([[0, 0, -1], [0, 1, -1], [1, 0, -1]])

        result = np.array_equal(self.valuation.cheapest_for_bundles, expected)
        self.assertTrue(result)

    def test_storage_2(self):
        item_values = np.array([[0, 9, 6, 7, 1, 1],
                                [2, 7, 1, 7, 10, 2],
                                [4, 1, 9, 2, 5, 4]])
        bundle_0 = structures.Bundle([2, 5], 0)
        bundle_1 = structures.Bundle([], 1)
        bundle_2 = structures.Bundle([], 2)
        allocation = structures.Allocation([bundle_0, bundle_1, bundle_2])
        valuation = structures.Valuation(item_values)
        valuation.initialize_bundle_values(allocation)
        valuation.initialize_cheap_lookup(allocation)

        index_0 = valuation.index_least_valuable_item(0, bundle_0)
        index_1 = valuation.index_least_valuable_item(1, bundle_0)

        item_0 = valuation.find_least_valuable_item(0, bundle_0)
        item_1 = valuation.find_least_valuable_item(1, bundle_0)

        expected_bundle_values = np.array([[7, 0, 0],
                                           [3, 0, 0],
                                           [13, 0, 0]])
        expected_cheapest = np.array([[1, -1, -1],
                                      [0, -1, -1],
                                      [1, -1, -1]])
        expected_index_0 = 1
        expected_index_1 = 0
        expected_item_0 = 5
        expected_item_1 = 2

        result_bundle_values = np.array_equal(expected_bundle_values, valuation.bundle_values)
        result_cheapest = np.array_equal(expected_cheapest, valuation.cheapest_for_bundles)
        self.assertTrue(result_bundle_values)
        self.assertTrue(result_cheapest)
        self.assertEqual(expected_index_0, index_0)
        self.assertEqual(expected_index_1, index_1)
        self.assertEqual(expected_item_0, item_0)
        self.assertEqual(expected_item_1, item_1)

    def test_find_least_valuable_item(self):
        item = self.valuation.find_least_valuable_item(self.agent, self.bundle)

        result = item == 2 or item == 4
        self.assertTrue(result)

    def test_valuate_bundle_item_removed(self):
        reduced_value = self.valuation.valuate_bundle_item_removed(self.agent, self.bundle)

        expected_value = 10 + 5
        self.assertEqual(expected_value, reduced_value)

    def test_remove_min_val_item_from_bundle(self):
        self.valuation.remove_min_val_item_from_bundle(self.agent, self.bundle)
        expected_bundle_items_option1 = [0, 2]  # remove either item 2 or 4
        expected_bundle_items_option2 = [0, 4]
        result1 = same_contents(self.bundle.items, expected_bundle_items_option1)
        result2 = same_contents(self.bundle.items, expected_bundle_items_option2)
        result = result1 or result2
        self.assertTrue(result)

    def test_is_bundle_efx_feasible(self):
        vals = np.array([[10, 9, 4, 6], [10, 6, 9, 4], [10, 4, 6, 9]])
        valuation = structures.Valuation(vals)
        bundle_0 = structures.Bundle([0], 0)
        bundle_1 = structures.Bundle([1, 3], 1)
        bundle_2 = structures.Bundle([2], 2)
        allocation = structures.Allocation([bundle_0, bundle_1, bundle_2])
        valuation.initialize_bundle_values(allocation)
        valuation.initialize_cheap_lookup(allocation)

        result0_0 = valuation.is_bundle_efx_feasible_for_agent(0, bundle_0, allocation)
        self.assertTrue(result0_0)

        result0_2 = valuation.is_bundle_efx_feasible_for_agent(0, bundle_2, allocation)
        self.assertFalse(result0_2)

    def test_overrule_preferred(self):
        vals = np.array([[10, 9, 4, 6], [10, 6, 9, 4], [10, 4, 6, 9]])
        valuation = structures.Valuation(vals)
        bundle_0 = structures.Bundle([0], 0)
        bundle_1 = structures.Bundle([1, 3], 1)
        bundle_2 = structures.Bundle([2], 2)
        allocation = structures.Allocation([bundle_0, bundle_1, bundle_2])
        valuation.initialize_bundle_values(allocation)
        valuation.initialize_cheap_lookup(allocation)

        result2_0 = valuation.does_bundle_overrule_preferred(2, bundle_0, allocation)
        self.assertTrue(result2_0)

        result1_0 = valuation.does_bundle_overrule_preferred(1, bundle_0, allocation)
        self.assertFalse(result1_0)

    def test_robust_demand(self):
        vals = np.array([[5, 9, 4, 6, 3, 5], [6, 6, 9, 4, 4, 7], [6, 4, 6, 9, 7, 3]])
        valuation = structures.Valuation(vals)
        bundle_0 = structures.Bundle([3, 5], 0)
        bundle_1 = structures.Bundle([0, 4], 1)
        bundle_2 = structures.Bundle([1, 2], 2)
        allocation = structures.Allocation([bundle_0, bundle_1, bundle_2])
        valuation.initialize_bundle_values(allocation)
        valuation.initialize_cheap_lookup(allocation)

        result_0 = valuation.robust_demand(0, allocation)
        self.assertEqual(result_0, bundle_2)

        result_1 = valuation.robust_demand(1, allocation)
        self.assertEqual(result_1, bundle_2)

        result_2 = valuation.robust_demand(2, allocation)
        self.assertEqual(result_2, bundle_0)

    def test_efx_check(self):
        valuations = structures.Valuation(np.array([[2, 2, 2, 3, 3, 3, 4],
                                                    [6, 5, 4, 3, 2, 1, 0],
                                                    [0, 1, 2, 3, 4, 5, 6]]))

        y_0 = structures.Bundle([6], 0)
        y_1 = structures.Bundle([2, 3], 1)
        y_2 = structures.Bundle([4], 2)
        alloc_1 = structures.Allocation([y_0, y_1, y_2])
        valuations.initialize_bundle_values(alloc_1)
        valuations.initialize_cheap_lookup(alloc_1)

        result_1 = valuations.is_allocation_efx(alloc_1)

        self.assertTrue(result_1)

        y_3 = structures.Bundle([6, 2], 0)
        y_4 = structures.Bundle([3], 1)
        y_5 = structures.Bundle([4], 2)
        alloc_2 = structures.Allocation([y_3, y_4, y_5])
        valuations.initialize_bundle_values(alloc_2)
        valuations.initialize_cheap_lookup(alloc_2)

        result_2 = valuations.is_allocation_efx(alloc_2)

        self.assertFalse(result_2)

    def test_nash_welfare(self):
        valuations = structures.Valuation(np.array([[2, 2, 2, 3, 3, 3, 4],
                                                    [6, 5, 4, 3, 2, 1, 0],
                                                    [0, 1, 2, 3, 4, 5, 6]]))

        y_0 = structures.Bundle([6], 0)
        y_1 = structures.Bundle([2, 3], 1)
        y_2 = structures.Bundle([4], 2)
        alloc_1 = structures.Allocation([y_0, y_1, y_2])
        valuations.initialize_bundle_values(alloc_1)
        valuations.initialize_cheap_lookup(alloc_1)

        result_1 = valuations.calculate_nash_welfare(alloc_1)

        expected_1 = (4 * (4 + 3) * 4) ** (1 / 3)
        self.assertAlmostEqual(result_1, expected_1)

        y_3 = structures.Bundle([6, 2], 0)
        y_4 = structures.Bundle([3], 1)
        y_5 = structures.Bundle([4], 2)
        alloc_2 = structures.Allocation([y_3, y_4, y_5])
        valuations.initialize_bundle_values(alloc_2)
        valuations.initialize_cheap_lookup(alloc_2)

        result_2 = valuations.calculate_nash_welfare(alloc_2)

        expected_2 = ((4 + 2) * 3 * 4) ** (1 / 3)
        self.assertAlmostEqual(result_2, expected_2)


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.edges1 = np.array([[True, False, False], [False, True, True], [False, False, False]])
        self.g_1 = structures.Graph(3, self.edges1)

    def test_remove_edge(self):
        self.g_1.remove_edge(1, 2)

        expected_edges = np.array([[True, False, False], [False, True, False], [False, False, False]])
        result = np.array_equal(self.g_1.edges, expected_edges)
        self.assertTrue(result)

    def test_add_edge(self):
        self.g_1.add_edge(2, 0)

        expected_edges = np.array([[True, False, False], [False, True, True], [True, False, False]])
        result = np.array_equal(self.g_1.edges, expected_edges)
        self.assertTrue(result)


class TestEfxGraph(unittest.TestCase):
    def test_init_1(self):
        vals = np.array([[10, 9, 4, 6], [10, 6, 9, 4], [10, 4, 6, 9]])
        valuation = structures.Valuation(vals)
        bundle_0 = structures.Bundle([0, 1], 0)
        bundle_1 = structures.Bundle([2], 1)
        bundle_2 = structures.Bundle([3], 2)
        allocation = structures.Allocation([bundle_0, bundle_1, bundle_2])

        efx_graph = structures.EfxGraph(3, allocation, valuation)

        expected_efx_graph = np.array([[True, False, False], [True, False, False], [True, False, False]])
        result = np.array_equal(expected_efx_graph, efx_graph.edges)
        self.assertTrue(result)

    def test_init_2(self):
        vals = np.array([[10, 9, 4, 6], [10, 6, 9, 4], [10, 4, 6, 9]])
        valuation = structures.Valuation(vals)
        bundle_0 = structures.Bundle([0], 0)
        bundle_1 = structures.Bundle([1, 3], 1)
        bundle_2 = structures.Bundle([2], 2)
        allocation = structures.Allocation([bundle_0, bundle_1, bundle_2])

        efx_graph = structures.EfxGraph(3, allocation, valuation)

        expected_efx_graph = np.array([[True, True, False], [False, True, False], [True, True, False]])
        result = np.array_equal(expected_efx_graph, efx_graph.edges)
        self.assertTrue(result)


class TestMatching(unittest.TestCase):
    def test_weighted_graph_1(self):
        bundle_0 = structures.Bundle([0], 0)
        bundle_1 = structures.Bundle([1, 3], 1)
        bundle_2 = structures.Bundle([2], 2)
        allocation = structures.Allocation([bundle_0, bundle_1, bundle_2])
        edges: np.ndarray = np.array([[True, True, False], [False, True, False], [True, True, False]])

        weighted_graph = structures.Matching.weighted_graph(allocation, edges)

        expected_weights = np.array([[10, 1, 0], [0, 10, 0], [1, 1, 0]])
        self.assertTrue(np.array_equal(weighted_graph, expected_weights))

    def test_weighted_graph_2(self):
        bundle_0 = structures.Bundle([0], 0)
        bundle_0.touched = True
        bundle_1 = structures.Bundle([1, 3], 1)
        bundle_2 = structures.Bundle([2], 2)
        bundle_2.touched = True
        allocation = structures.Allocation([bundle_0, bundle_1, bundle_2])
        edges: np.ndarray = np.array([[True, True, False], [False, True, False], [True, True, False]])

        weighted_graph = structures.Matching.weighted_graph(allocation, edges)

        expected_weights = np.array([[91, 1, 0], [0, 10, 0], [82, 1, 0]])
        self.assertTrue(np.array_equal(weighted_graph, expected_weights))

    def test_init(self):
        bundle_0 = structures.Bundle([0], 0)
        bundle_1 = structures.Bundle([1, 3], 1)
        bundle_2 = structures.Bundle([2], 2)
        allocation = structures.Allocation([bundle_0, bundle_1, bundle_2])
        edges: np.ndarray = np.array([[True, True, False], [False, True, False], [True, True, False]])

        matching = structures.Matching(3, allocation, edges)

        expected_matching = np.array([[True, False, False], [False, True, False], [False, False, False]])
        self.assertTrue(np.array_equal(matching.edges, expected_matching))

    def test_init_identity(self):
        bundle_0 = structures.Bundle([0], 0)
        bundle_1 = structures.Bundle([1, 3], 1)
        bundle_2 = structures.Bundle([2], 2)
        allocation = structures.Allocation([bundle_0, bundle_1, bundle_2])

        identity_matching = structures.Matching(3, allocation, None)

        expected_matching = np.array([[True, False, False], [False, True, False], [False, False, True]])
        self.assertTrue(np.array_equal(identity_matching.edges, expected_matching))

    def test_get_unmatched_bundle(self):
        bundle_0 = structures.Bundle([0], 0)
        bundle_1 = structures.Bundle([1, 3], 1)
        bundle_2 = structures.Bundle([2], 2)
        allocation = structures.Allocation([bundle_0, bundle_1, bundle_2])
        edges: np.ndarray = np.array([[True, True, False], [False, True, False], [True, True, False]])
        matching = structures.Matching(3, allocation, edges)

        unmatched_bundle = matching.get_unmatched_bundle()

        expected_bundle = bundle_2
        self.assertEqual(unmatched_bundle, expected_bundle)

    def test_get_agent_match_1(self):
        bundle_0 = structures.Bundle([0], 0)
        bundle_1 = structures.Bundle([1, 3], 1)
        bundle_2 = structures.Bundle([2], 2)
        allocation = structures.Allocation([bundle_0, bundle_1, bundle_2])
        edges: np.ndarray = np.array([[True, True, False], [False, True, False], [True, True, False]])
        matching = structures.Matching(3, allocation, edges)

        agent = matching.get_agent_matched_to_bundle(bundle_1)

        expected_agent = 1
        self.assertEqual(agent, expected_agent)

    def test_get_agent_match_2(self):
        bundle_0 = structures.Bundle([0], 0)
        bundle_1 = structures.Bundle([1, 3], 1)
        bundle_2 = structures.Bundle([2], 2)
        allocation = structures.Allocation([bundle_0, bundle_1, bundle_2])
        edges: np.ndarray = np.array([[True, True, False], [False, True, False], [True, True, False]])
        matching = structures.Matching(3, allocation, edges)
        modified_edges = np.array([[False, True, False], [True, False, False], [False, False, True]])
        matching.edges = modified_edges

        agent = matching.get_agent_matched_to_bundle(bundle_1)

        expected_agent = 0
        self.assertEqual(agent, expected_agent)

    def test_get_bundle_match(self):
        bundle_0 = structures.Bundle([0], 0)
        bundle_1 = structures.Bundle([1, 3], 1)
        bundle_2 = structures.Bundle([2], 2)
        allocation = structures.Allocation([bundle_0, bundle_1, bundle_2])
        edges: np.ndarray = np.array([[True, True, False], [False, True, False], [True, True, False]])
        matching = structures.Matching(3, allocation, edges)
        modified_edges = np.array([[False, True, False], [True, False, False], [False, False, True]])
        matching.edges = modified_edges

        bundle = matching.get_bundle_matched_to_agent(0)

        expected_bundle = bundle_1
        self.assertEqual(bundle, expected_bundle)

    def test_get_result_allocation(self):
        bundle_0 = structures.Bundle([0], 0)
        bundle_1 = structures.Bundle([1, 3], 1)
        bundle_2 = structures.Bundle([2], 2)
        allocation = structures.Allocation([bundle_0, bundle_1, bundle_2])
        edges: np.ndarray = np.array([[True, True, False], [False, True, False], [True, True, False]])
        matching = structures.Matching(3, allocation, edges)
        modified_edges = np.array([[False, True, False], [True, False, False], [False, False, True]])
        matching.edges = modified_edges
        matching.size = 3

        result_allocation = matching.get_result_allocation(allocation)

        expected_allocation_bundles = np.array([bundle_1, bundle_0, bundle_2])
        self.assertTrue(np.array_equal(result_allocation.bundles, expected_allocation_bundles))

    def test_is_agent_matched(self):
        bundle_0 = structures.Bundle([0], 0)
        bundle_1 = structures.Bundle([1, 3], 1)
        bundle_2 = structures.Bundle([2], 2)
        allocation = structures.Allocation([bundle_0, bundle_1, bundle_2])
        edges: np.ndarray = np.array([[True, True, False], [False, True, False], [True, True, False]])
        matching = structures.Matching(3, allocation, edges)

        is_agent_0_matched = matching.is_agent_matched(0)
        self.assertTrue(is_agent_0_matched)

        is_agent_2_matched = matching.is_agent_matched(2)
        self.assertFalse(is_agent_2_matched)


class TestAugmentingPath(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle_0 = structures.Bundle([0], 0)
        self.bundle_1 = structures.Bundle([1, 3], 1)
        self.bundle_2 = structures.Bundle([2], 2)
        self.bundle_3 = structures.Bundle([], 3)
        self.bundle_4 = structures.Bundle([], 4)

        allocation = structures.Allocation([self.bundle_0, self.bundle_1, self.bundle_2])
        edges: np.ndarray = np.array([[True, True, False], [False, True, False], [True, True, False]])
        m = structures.Matching(3, allocation, edges)
        m_edges = np.array([[False, False, True, False, False],
                            [False, False, False, False, True],
                            [False, True, False, False, False],
                            [False, False, False, False, False],
                            [False, False, False, False, False]])
        alloc = structures.Allocation([self.bundle_0, self.bundle_1, self.bundle_2, self.bundle_3, self.bundle_4])
        m.edges = m_edges
        m.agents = 5
        m.size = 3
        m.allocation = alloc

        m0 = structures.Matching(5, allocation, None)

        self.path = structures.AugmentingPath(self.bundle_0, m, m0)

    def test_init_1(self):
        expected_path = np.array([[0, self.bundle_2], [2, self.bundle_1], [1, self.bundle_4]])
        self.assertTrue(np.array_equal(self.path.path, expected_path))
        self.assertEqual(self.path.jk, 4)

    def test_in_path(self):
        z0_in_path = self.path.is_bundle_in_path(self.bundle_0)
        self.assertFalse(z0_in_path)

        z2_in_path = self.path.is_bundle_in_path(self.bundle_2)
        self.assertTrue(z2_in_path)


class TestAllocator(unittest.TestCase):
    def test_x_hat(self):
        x_0 = structures.Bundle([0, 1, 2, 3, 4], 0)
        x_1 = structures.Bundle([5, 6], 1)
        x_2 = structures.Bundle([], 2)
        x = structures.Allocation([x_0, x_1, x_2])

        z_0 = structures.Bundle([0, 1, 2, 3], 0)
        z_1 = structures.Bundle([5, 6], 1)
        z_2 = structures.Bundle([], 2)
        z = structures.Allocation([z_0, z_1, z_2])

        p = Mock(structures.AugmentingPath)
        p.jk = 1
        path = [(2, z_1)]
        p.path = np.array(path)
        z_jstar = z_0

        x_hat = allocator.Allocator.compute_x_hat(x, z, p, z_jstar)

        hat_0_items = np.array([4])
        hat_1_items = np.array([0, 1, 2, 3])
        hat_2_items = np.array([5, 6])
        self.assertTrue(np.array_equal(x_hat.bundles[0].items, hat_0_items))
        self.assertTrue(np.array_equal(x_hat.bundles[1].items, hat_1_items))
        self.assertTrue(np.array_equal(x_hat.bundles[2].items, hat_2_items))


if __name__ == '__main__':
    unittest.main()
