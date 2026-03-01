import random

import dendropy
from dendropy.simulate import treesim

# Parameters
n_taxa = 1000
birth_rate = 1.0
death_rate = 0.3
target_height = 1.0
rate_min = 0.5
rate_max = 2.0

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


# Save tree
tree.write(
    path="bd_1000taxa_fixedheight.nwk",
    schema="newick"
)

print("Tree simulated.")
print("Final height:", tree_height(tree))
print("Number of taxa:", len(tree.leaf_nodes()))
print("newick:", tree)
