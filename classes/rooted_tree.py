import copy
from classes.node import Node
from classes.unrooted_tree import UnrootedTree
from enums import RootingMethods


class RootedTree:
    root: Node
    all_nodes: list[Node]
    keys: set[str]
    seq_weight_dict: dict[str, float]

    def __init__(self, root: Node, all_nodes: list[Node], keys: set[str]):
        self.root = root
        self.all_nodes = all_nodes
        self.keys = keys
        self.seq_weight_dict = {}

    @classmethod
    def root_tree(cls, unrooted: UnrootedTree, rooting_method: RootingMethods):
        all_nodes: list[Node] = copy.deepcopy(unrooted.all_nodes)
        keys: set[str] = copy.copy(unrooted.keys)
        new_root: Node
        if rooting_method == RootingMethods.LONGEST_PATH_MID:
            rooting_point = calc_mid_point(unrooted)
        else:
            rooting_points = calc_min_differential_sum(unrooted, all_nodes)
            rooting_point = find_shallowest_tree(all_nodes, rooting_points)
        new_root, all_nodes = create_root(all_nodes, rooting_point)
        return cls(root=new_root, all_nodes=all_nodes, keys=keys)

    def calc_clustal_w(self):
        nodes_to_recalc: list[Node] = [self.root]
        while len(nodes_to_recalc) > 0:
            node = nodes_to_recalc.pop(0)
            fill_nodes_w(node, nodes_to_recalc)

    def calc_seq_w(self):
        self.calc_clustal_w()
        for node in self.all_nodes:
            if len(node.children) == 0:
                self.seq_weight_dict[list(node.keys)[0]] = node.weight


def calc_mid_point(unrooted: UnrootedTree) -> dict:
    path, max_dist = unrooted.longest_path()
    half_length: float = 0
    for b in path:
        if half_length + b['dist'] > max_dist / 2:
            dist_from_start = max_dist / 2 - half_length
            return {'start_id': b['start_id'],
                    'end_id': b['end_id'],
                    'dist_from_start': dist_from_start,
                    'dist_from_end':  b['dist'] - dist_from_start}
        else:
            half_length += b['dist']

def calc_min_differential_sum(unrooted: UnrootedTree, all_nodes: list[Node]) -> list[dict]:
    all_branches: dict[str, dict] = {}
    all_bl: list[float] = unrooted.get_branches_lengths_list()
    total_branch_length: float = sum(all_bl)
    min_bl: float = min(all_bl)
    for node in all_nodes:
        for n in node.get_adj():
            min_id = min(n['node'].id, node.id)
            max_id = max(n['node'].id, node.id)
            key: str = f'o:{min_id}-d:{max_id}'
            if key not in all_branches:
                this_branch = {'origin': min_id, 'dest': max_id, 'bl': n['dist'],
                               'w_to_orig': sum_bl_up_to_node_id(all_nodes[min_id], max_id)}
                this_branch['w_to_dest'] = total_branch_length - this_branch['w_to_orig'] - this_branch['bl']
                this_branch['delta'] = this_branch['bl'] - abs(this_branch['w_to_dest'] - this_branch['w_to_orig'])
                all_branches[key] = this_branch
    res: list[dict] = [calc_potential_root_on_branch(b, min_bl) for b in all_branches.values() if b['delta'] > 0]
    if len(res) == 0:
        res = [calc_potential_root_on_branch(sorted(all_branches.values(), key=lambda x: x['delta'], reverse=True)[0],
                                             min_bl)]
    return res

def recalc_tree_down(node: Node, father: Node, broke_id: int, nodes_to_recalc: list[dict], all_nodes: list[Node]):
    node.set_rank_from_root(father.rank_from_root + 1)
    if father in node.children:
        node.branch_length = father.branch_length
    adj = node.get_adj()
    children = [all_nodes[n['node'].id] for n in adj if (father is None or n['node'].id != father.id) and n['node'].id != broke_id]
    if len(children):
        node.update_children_only(children)
        nodes_to_recalc.append({'node': children[0], 'father': node, 'broke': broke_id})
        nodes_to_recalc.append({'node': children[1], 'father': node, 'broke': broke_id})
    node.father = father

def fill_nodes_w(node: Node, nodes_to_recalc: list[Node]):
    node.set_w_from_root((node.father.w_from_root.copy() if node.father is not None else []) + [node.branch_length])
    if len(node.children):
        nodes_to_recalc.append(node.children[0])
        nodes_to_recalc.append(node.children[1])
    else:
        node.set_weight_from_root()


def sum_bl_up_to_node_id(origin: Node, dest_id: int) -> float:
    w_to_orig: float = 0
    visited_node_ids: set[int] = {dest_id, origin.id}
    queue: list[dict] = [data for data in origin.get_adj() if data['node'].id not in visited_node_ids]
    while len(queue) > 0:
        next_node = queue.pop()
        if next_node['node'].id not in visited_node_ids:
            w_to_orig += next_node['dist']
            visited_node_ids.add(next_node['node'].id)
            queue += [data for data in next_node['node'].get_adj() if data['node'].id not in visited_node_ids]
    return w_to_orig

def calc_potential_root_on_branch(b: dict, min_bl: float) -> dict:
    min_bl /= 10
    if b['delta'] > 0:
        dist_from_start: float = b['bl'] / 2 - (b['w_to_orig'] - b['w_to_dest']) / 2
    elif b['w_to_orig'] > b['w_to_dest']:
        dist_from_start: float = min_bl
    else:
        dist_from_start: float = b['bl'] - min_bl
    return {'start_id': b['origin'], 'end_id': b['dest'], 'dist_from_start': dist_from_start,
            'dist_from_end': b['bl'] - dist_from_start}

def create_root(all_nodes: list[Node], rooting_point: dict) -> tuple[Node, list[Node]]:
    new_root_id: int = len(all_nodes)
    start_id: int = rooting_point['start_id']
    end_id: int = rooting_point['end_id']
    new_root = Node(node_id=new_root_id, keys=set(),
                    children=[all_nodes[start_id], all_nodes[end_id]],
                    children_bl_sum=0)
    new_root.set_rank_from_root(0)
    all_nodes.append(new_root)
    nodes_to_recalc: list[dict] = [{'node': all_nodes[start_id], 'father': new_root, 'broke': end_id},
                                   {'node': all_nodes[end_id], 'father': new_root, 'broke': start_id}]
    while len(nodes_to_recalc):
        data = nodes_to_recalc.pop()
        recalc_tree_down(data['node'], data['father'], data['broke'], nodes_to_recalc, all_nodes)
    all_nodes[start_id].set_branch_length(rooting_point['dist_from_start'])
    all_nodes[end_id].set_branch_length(rooting_point['dist_from_end'])

    nodes_sorted_by_rank: list[Node] = sorted(all_nodes, key=lambda x: x.rank_from_root)
    while len(nodes_sorted_by_rank):
        node_to_update = nodes_sorted_by_rank.pop()
        node_to_update.update_data_from_children()
    return new_root, all_nodes


def find_shallowest_tree(all_nodes: list[Node], rooting_points: list[dict]) -> dict:
    if len(rooting_points) == 1:
        return rooting_points[0]
    for rp in rooting_points:
        start_id: int = rp['start_id']
        end_id: int = rp['end_id']
        my_nodes = copy.deepcopy(all_nodes)
        if my_nodes[start_id].father.id == end_id:
            add_node_between(my_nodes, my_nodes[start_id], my_nodes[end_id],
                             rp['dist_from_start'], rp['dist_from_end'])
        else:
            add_node_between(my_nodes, my_nodes[end_id], my_nodes[start_id],
                             rp['dist_from_end'], rp['dist_from_start'])
        unrooted_tree = UnrootedTree(my_nodes[-2], my_nodes)
        rp['longest_dist'] = unrooted_tree.get_longest_dist_to(my_nodes[-1])
    rooting_points.sort(key=lambda x: x['longest_dist'])
    return rooting_points[0]

def add_node_between(my_nodes: list[Node], child: Node, father: Node, dist_from_child: float,
                     dist_to_father: float):
    new_node = Node.create_from_children([child], len(my_nodes))
    new_node.set_a_father(father)
    new_node.set_branch_length(dist_to_father)
    father.children = [c for c in father.children if c.id != child.id] + [new_node]
    child.set_a_father(new_node)
    child.set_branch_length(dist_from_child)
    my_nodes.append(new_node)





