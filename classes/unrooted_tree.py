from pathlib import Path
from classes.node import Node

EPSILON = 0.001


class UnrootedTree:
    anchor: Node
    all_nodes: list[Node]
    keys: set[str]
    differentiator_key: str

    def __init__(self, anchor: Node, all_nodes: list[Node]):
        self.anchor = anchor
        self.all_nodes = all_nodes
        self.keys = set(anchor.keys)
        self.differentiator_key = sorted(list(self.keys))[0]

    @classmethod
    def create_from_newick_file(cls, path: Path):
        newick_str: str = read_newick_from_file(path)
        anchor, all_nodes = root_from_newick_str(newick_str)
        return cls(anchor=anchor, all_nodes=all_nodes)

    @classmethod
    def create_from_newick_str(cls, newick_str: str):
        anchor, all_nodes = root_from_newick_str(newick_str)
        return cls(anchor=anchor, all_nodes=all_nodes)

    def get_internal_edges_set(self) -> set[str]:
        edges: set[str] = set()
        for n in self.all_nodes:
            if len(n.children) > 0:
                edges_str: str | None = n.get_keys_unrooted_string(self.keys, self.differentiator_key)
                if edges_str is not None:
                    edges.add(edges_str)
        return edges

    def calc_rf(self, other_tree: 'UnrootedTree'):
        return len(self.get_internal_edges_set() ^ other_tree.get_internal_edges_set())

    def get_branches_lengths_list(self) -> list[float]:
        bl_list: list[float] = []
        for n in self.all_nodes:
            if n.father is not None:
                bl_list.append(max(n.branch_length, EPSILON))
        return bl_list

    def get_longest_path(self, u: Node) -> tuple[Node, list[dict], float]:
        nodes_count: int = len(self.all_nodes)
        nodes_by_id: list[Node | None] = [None for i in range(nodes_count + 1)]
        # mark all distance with -1
        distance = [-1 for i in range(nodes_count + 1)]
        path: list[list[dict]] = [[] for i in range(nodes_count + 1)]

        # distance of u from u will be 0
        distance[u.id] = 0
        path[u.id] = []
        # in-built library for queue which performs fast operations on both the ends
        queue: list[Node] = [u]
        nodes_by_id[u.id] = u
        # mark node u as visited

        while len(queue) > 0:

            # pop the front of the queue(0th element)
            front = queue.pop(0)
            # loop for all adjacent nodes of node front

            for i in front.get_adj():

                if nodes_by_id[i['node'].id] is None:
                    # mark the ith node as visited
                    nodes_by_id[i['node'].id] = i['node']
                    # make distance of i , one more than distance of front
                    distance[i['node'].id] = distance[front.id] + i['dist']
                    path[i['node'].id] = path[front.id].copy()
                    path[i['node'].id].append({'start_id': front.id, 'end_id': i['node'].id, 'dist': i['dist']})
                    # Push node into the stack only if it is not visited already
                    queue.append(i['node'])

        max_dist: float = 0
        # get farthest node distance and its index
        node_index: int = -1
        for i in range(nodes_count):
            if distance[i] > max_dist:
                max_dist = distance[i]
                node_index = i

        return nodes_by_id[node_index], path[node_index], max_dist

    def longest_path(self) -> tuple[list[dict], float]:

        # first DFS to find one end point of longest path
        node, path, max_dist = self.get_longest_path(self.all_nodes[0])

        # second DFS to find the actual longest path
        node_2, path, max_dist = self.get_longest_path(node)
        # print('Longest path is:', path)
        return path, max_dist

    def get_longest_dist_to(self, dest: Node) -> float:
        nodes_count: int = len(self.all_nodes)
        nodes_by_id: list[Node | None] = [None for i in range(nodes_count + 1)]
        distance = [-1 for i in range(nodes_count + 1)]
        distance[dest.id] = 0
        queue: list[Node] = [dest]
        nodes_by_id[dest.id] = dest
        while len(queue) > 0:
            front = queue.pop(0)
            for i in front.get_adj():
                if nodes_by_id[i['node'].id] is None:
                    nodes_by_id[i['node'].id] = i['node']
                    distance[i['node'].id] = distance[front.id] + i['dist']
                    queue.append(i['node'])
        return max(distance)


def root_from_newick_str(newick_str: str) -> tuple[Node, list[Node]]:
    root, all_nodes = create_a_tree_from_newick(newick_str)
    if len(root.children) == 3:
        return root, all_nodes
    if len(root.children) == 2:
        res: list[Node] = []
        root.children.sort(key=lambda x: x.branch_length)
        root.children.sort(key=lambda x: len(x.children), reverse=True)
        if len(root.children[0].children) == 2:
            res += root.children[0].children
            res.append(root.children[1])
            anchor = Node.create_from_children(children_list=res, inx=len(all_nodes))
            all_nodes.append(anchor)
            return anchor, all_nodes


def read_newick_from_file(input_file_path: Path) -> str:
    with open(input_file_path, 'r') as in_file:
        for line in in_file:
            return line.strip()


def create_a_tree_from_newick(newick: str) -> tuple[Node, list[Node]]:
    all_nodes: list[Node] = []
    open_nodes_per_level: dict[int, list[Node]] = {}
    level: int = 0
    current_key: str = ''
    branch_length: str = ''
    i: int = 0
    while i < len(newick):
        if newick[i] == '(':
            level += 1
            if level not in open_nodes_per_level:
                open_nodes_per_level[level] = []
            i += 1
        elif newick[i] == ':':
            branch_length = ''
            i += 1
            while newick[i] != ')' and newick[i] != ',':
                branch_length += newick[i]
                i += 1
        elif newick[i] == ',' or newick[i] == ')':
            if len(current_key) > 0:
                current_node = Node(node_id=len(all_nodes), keys={current_key}, children=[], children_bl_sum=0,
                                    branch_length=float(branch_length))
                open_nodes_per_level[level].append(current_node)
                current_key = ''
            else:
                current_node = create_node_from_children(open_nodes_per_level, level, float(branch_length),
                                                         len(all_nodes))
                open_nodes_per_level[level + 1] = []
                open_nodes_per_level[level].append(current_node)
            all_nodes.append(current_node)
            if newick[i] == ')':
                level -= 1
            i += 1
        elif newick[i] == ';':
            current_node = create_node_from_children(open_nodes_per_level, 0, float(branch_length),
                                                     len(all_nodes))
            return current_node, all_nodes
        else:
            current_key = ''
            while newick[i] != ')' and newick[i] != ',' and newick[i] != ':':
                current_key += newick[i]
                i += 1


def create_node_from_children(open_nodes_per_level: dict[int, list[Node]], level: int, branch_length: float,
                              node_inx: int) -> Node:
    node_keys = set()
    for child in open_nodes_per_level[level + 1]:
        node_keys = node_keys.union(child.keys)
    current_node = Node(node_id=node_inx, keys=node_keys, children=open_nodes_per_level[level + 1].copy(),
                        children_bl_sum=0, branch_length=float(branch_length))
    for child in open_nodes_per_level[level + 1]:
        child.set_a_father(current_node)
    return current_node




