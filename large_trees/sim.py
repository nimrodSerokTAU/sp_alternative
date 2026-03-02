import random
from pathlib import Path
from dendropy.simulate import treesim

# Parameters
birth_rate = 1.0
target_height = 1.0
rate_min = 0.5
rate_max = 2.0


def create_birth_death_tree(n_taxa: int, death_rate: float) -> str:
    # Simulate birth–death tree conditioned on number of extant taxa
    tree = treesim.birth_death_tree(
        birth_rate=birth_rate,
        death_rate=death_rate,
        num_extant_tips=n_taxa
    )

    # Compute current tree height (root-to-tip distance)
    def tree_height(tree):
        root = tree.seed_node
        return tree.max_distance_from_root()

    current_height = tree_height(tree)

    # Scale tree to fixed height
    scaling_factor = target_height / current_height
    for edge in tree.postorder_edge_iter():
        if edge.length is not None:
            edge.length *= scaling_factor

    for edge in tree.postorder_edge_iter():
        if edge.length is not None:
            rate_multiplier = random.uniform(rate_min, rate_max)
            edge.length *= rate_multiplier

    tree.write(
        path="bd_1000taxa_relaxed_uniform_0.5_2.nwk",
        schema="newick"
    )

    print("Final height:", tree_height(tree))
    print("newick:", tree)
    return tree.as_string(schema="newick", suppress_rooting=True)


def create_trees(trees_num: int, taxa_num: int, min_death_rate: float, max_death_rate: float, res_path: str):
    curr_dir = Path(__file__).parent
    file_path = curr_dir / res_path
    with open(file_path, "a") as file:
        for t in range(trees_num):
            death_rate = random.uniform(min_death_rate, max_death_rate)
            newick_str = create_birth_death_tree(taxa_num, death_rate)
            file.write(newick_str)


create_trees(70, 40, 0.3, 0.5, '../output/trees_40.txt')
create_trees(70, 50, 0.3, 0.5, '../output/trees_50.txt')
create_trees(70, 60, 0.3, 0.5, '../output/trees_60.txt')
create_trees(70, 70, 0.3, 0.5, '../output/trees_70.txt')
create_trees(70, 80, 0.3, 0.5, '../output/trees_80.txt')
