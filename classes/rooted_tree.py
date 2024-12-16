import copy
from classes.node import Node
from classes.unrooted_tree import UnrootedTree
from enums import RootingMethod


class RootedTree:
    root: Node
    all_nodes: list[Node]
    keys: set[str]

    def __init__(self, root: Node, all_nodes: list[Node], keys: set[str]):
        self.root = root
        self.all_nodes = all_nodes
        self.keys = keys

    @classmethod
    def root_tree(cls, unrooted: UnrootedTree, rooting_method: RootingMethod):
        all_nodes: list[Node] = copy.deepcopy(unrooted.all_nodes) # TODO: consider not using deep copy as the nodes will not refer to other nodes. consider moving over nodes and reattach.
        keys: set[str] = copy.copy(unrooted.keys)
        new_root: Node
        if rooting_method == RootingMethod.LONGEST_PATH_MID:
            start_id, end_id, dist_from_start, dist_from_end = calc_mid_point(unrooted)
            new_root_id = len(all_nodes)
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
            all_nodes[start_id].set_branch_length(dist_from_start)
            all_nodes[end_id].set_branch_length(dist_from_end)

        nodes_sorted_by_rank: list[Node] = sorted(all_nodes, key=lambda x: x.rank_from_root)
        while len(nodes_sorted_by_rank):
            node_to_update = nodes_sorted_by_rank.pop()
            node_to_update.update_data_from_children()
        return cls(root=new_root, all_nodes=all_nodes, keys=keys)

def calc_mid_point(unrooted: UnrootedTree) -> tuple[int, int, float, float]:
    path, max_dist = unrooted.longest_path()
    half_length: float = 0
    for b in path:
        if half_length + b['dist'] > max_dist / 2:
            dist_from_start = max_dist / 2 - half_length
            return b['start_id'], b['end_id'], dist_from_start, b['dist'] - dist_from_start
        else:
            half_length += b['dist']


def recalc_tree_down(node, father: Node, broke_id: int, nodes_to_recalc: list[dict], all_nodes: list[Node]):
    if node.id == 6:
        stop = True
    node.set_rank_from_root(father.rank_from_root + 1)
    adj = node.get_adj()
    children = [all_nodes[n['node'].id] for n in adj if (father is None or n['node'].id != father.id) and n['node'].id != broke_id]
    if len(children):
        node.update_children_only(children)
        nodes_to_recalc.append({'node': children[0], 'father': node, 'broke': broke_id})
        nodes_to_recalc.append({'node': children[1], 'father': node, 'broke': broke_id})
    node.father = father



