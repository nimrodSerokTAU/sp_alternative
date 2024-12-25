import copy
import sys
from typing import List

from classes.node import Node
from classes.unrooted_tree import UnrootedTree


class NeighborJoining:
    distance_matrix: List[List[float]]
    nodes: List[Node]
    q_matrix: List[List[float]]
    all_nodes: List[Node]
    tree_res: UnrootedTree

    def __init__(self, distanceMatrix: List[List[float]], nodes: List[Node]):
        self.distance_matrix = copy.deepcopy(distanceMatrix)
        self.nodes = nodes
        self.all_nodes = nodes.copy()
        self.tree_res = self.build_tree()

    def calc_q_matrix(self) -> List[List[int]]:
        number_of_seq = len(self.nodes)
        q_matrix = []
        for i in range(len(self.distance_matrix) - 1):
            q_matrix.append([0] * number_of_seq)
            for j in range(i + 1, len(self.distance_matrix)):
                q_matrix[i][j] = ((number_of_seq - 2) * self.distance_matrix[i][j] -
                                  sum(self.distance_matrix[i]) - sum(self.distance_matrix[j]))
        return q_matrix

    def find_closest_pair(self) -> tuple[int, int, float]:
        f_inx: int = -1
        s_inx: int = -1
        min_dist = sys.maxsize
        for i in range(len(self.q_matrix)):
            for j in range(i + 1, len(self.q_matrix)):
                if self.q_matrix[i][j] < min_dist:
                    min_dist: float = self.q_matrix[i][j]
                    f_inx = i
                    s_inx = j
        return f_inx, s_inx, min_dist

    def merge_two_clusters(self):
        f_inx, s_inx, min_dist = self.find_closest_pair()
        delta_f, delta_s = self.find_delta(f_inx, s_inx)
        matrix: list[list[float]] = []
        for r in range(len(self.distance_matrix)):
            row = copy.copy(self.distance_matrix[r])
            if r == f_inx:
                row[f_inx] = 0
            else:
                row[f_inx] = (row[f_inx] + row[s_inx] - self.distance_matrix[f_inx][s_inx]) / 2
            del row[s_inx]
            if r != s_inx:
                matrix.append(row)
            else:
                for n in range(len(row)):
                    if n != f_inx:
                        matrix[f_inx][n] = (matrix[f_inx][n] + row[n] - self.distance_matrix[f_inx][s_inx]) / 2

        self.distance_matrix = matrix
        self.nodes[f_inx].set_branch_length(delta_f)
        self.nodes[s_inx].set_branch_length(delta_s)
        new_node = Node.create_from_children([self.nodes[f_inx], self.nodes[s_inx]], len(self.all_nodes))
        # new_node.fill_newick()  # TODO: remove later for test only
        self.all_nodes.append(new_node)
        self.nodes[f_inx].set_a_father(new_node)
        self.nodes[s_inx].set_a_father(new_node)
        self.nodes[f_inx] = new_node

        del self.nodes[s_inx]

    def find_delta(self, f_inx: int, s_inx: int) -> tuple[float, float]:
        number_of_seq = len(self.nodes)
        delta_f: float = (0.5 * self.distance_matrix[f_inx][s_inx] +
                          (sum(self.distance_matrix[f_inx]) - sum(self.distance_matrix[s_inx])) / (
                                      2 * (number_of_seq - 2)))
        delta_s: float = self.distance_matrix[f_inx][s_inx] - delta_f
        return delta_f, delta_s

    def merge_last_three(self) -> Node:
        delta_f, delta_s = self.find_delta(0, 1)
        delta_t = self.distance_matrix[0][2] - delta_f
        deltas = [delta_f, delta_s, delta_t]
        for i in range(3):
            self.nodes[i].set_branch_length(deltas[i])
        new_node = Node.create_from_children([self.nodes[0], self.nodes[1], self.nodes[2]], len(self.all_nodes))
        self.all_nodes.append(new_node)
        for i in range(3):
            self.nodes[i].set_a_father(new_node)
        return new_node

    def build_tree(self) -> UnrootedTree:
        while len(self.nodes) > 3:
            self.q_matrix = self.calc_q_matrix()
            self.merge_two_clusters()
        self.q_matrix = self.calc_q_matrix()
        root = self.merge_last_three()
        return UnrootedTree(root, self.all_nodes)
