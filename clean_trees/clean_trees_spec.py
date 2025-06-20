from clean_trees import clean_nodes, read_file_and_get_names, remove_some_leaves, \
    get_length_from_root_from_newick_and_name


def test_tree_case_a():
    tree_str = '((((H:1,K:1)1:1,(F:1,I:1)1:1)1:1,E:1)1:1,((L:1,(N:1,Q:1)1:1)1:1,(P:1,S:1)1:1)1:1);'
    res, names = clean_nodes(tree_str, ['H', 'K', 'I', 'E', 'L', 'N', 'Q', 'S'])
    assert res == '((((H:1,K:1)1:1,I:2)1:1,E:1)1:1,((L:1,(N:1,Q:1)1:1)1:1,S:2)1:1);'


def test_tree_case_b():
    tree_str = '((((H:1,K:1)1:1,(F:1,I:1)1:1)1:1,E:1)1:1,((L:1,(N:1,Q:1)1:1)1:1,(P:1,S:1)1:1)1:1);'
    res, names = clean_nodes(tree_str, ['H', 'K', 'L', 'N', 'Q', 'S'])
    assert res == '(((L:1,(N:1,Q:1)1:1)1:1,S:2)1:1,(H:1,K:1)1:3);'
    bl_after = get_length_from_root_from_newick_and_name(res, 'H')
    assert bl_after == 5


def test_read_file_and_get_names():
    newick_str, leaf_names = read_file_and_get_names('100507096_no_supports.tree')
    bl_before = get_length_from_root_from_newick_and_name(newick_str, 'Mesocricetus_auratus')
    assert len(leaf_names) == 186
    assert leaf_names[0] == 'Mesocricetus_auratus'
    assert newick_str[0:50] == '(Mesocricetus_auratus:0.1322,(((((((((((((((((Chry'
    bl_after = get_length_from_root_from_newick_and_name(newick_str, 'Mesocricetus_auratus')
    assert bl_before == bl_after

def test_read_and_keep_15():
    newick_str, leaf_names = read_file_and_get_names('100507096_no_supports.tree')
    bl_before = get_length_from_root_from_newick_and_name(newick_str, 'Echinops_telfairi')
    keep_leafs = ['Mesocricetus_auratus', 'Echinops_telfairi', 'Elephantulus_edwardii', 'Orycteropus_afer_afer', 'Loxodonta_africana',  'Suricata_suricatta', 'Ursus_maritimus', 'Ursus_arctos', 'Ursus_americanus', 'Ailuropoda_melanoleuca', 'Halichoerus_grypus', 'Phoca_vitulina', 'Leptonychotes_weddellii', 'Neomonachus_schauinslandi', 'Mirounga_leonina']
    res_tree, leaf_names = clean_nodes(newick_str, keep_leafs)
    assert len(leaf_names) == 15
    bl_after = get_length_from_root_from_newick_and_name(res_tree, 'Echinops_telfairi')
    assert bl_before == bl_after


def test_read_and_keep_50():
    newick_str, leaf_names = read_file_and_get_names('100507096_no_supports.tree')
    keep_leafs = ['Mesocricetus_auratus', 'Chrysochloris_asiatica', 'Elephantulus_edwardii', 'Orycteropus_afer_afer',
                   'Elephas_maximus_indicus', 'Phascolarctos_cinereus', 'Antechinus_flavipes', 'Sarcophilus_harrisii',
                   'Gracilinanus_agilis', 'Dasypus_novemcinctus', 'Condylura_cristata', 'Sorex_araneus',
                   'Delphinapterus_leucas', 'Monodon_monoceros', 'Phocoena_sinus', 'Lagenorhynchus_obliquidens',
                   'Ovis_aries', 'Capra_hircus', 'Oryx_dammah', 'Odocoileus_virginianus_texanus', 'Camelus_bactrianus',
                   'Camelus_ferus', 'Camelus_dromedarius', 'Vicugna_pacos', 'Tupaia_chinensis', 'Macaca_nemestrina',
                   'Microcebus_murinus', 'Lemur_catta', 'Propithecus_coquereli', 'Carlito_syrichta',
                   'Chinchilla_lanigera', 'Octodon_degus', 'Cavia_porcellus', 'Jaculus_jaculus', 'Castor_canadensis',
                   'Marmota_marmota_marmota', 'Marmota_monax', 'Nannospalax_galili', 'Meriones_unguiculatus',
                   'Grammomys_surdaster', 'Mus_caroli', 'Mus_musculus', 'Mastomys_coucha', 'Peromyscus_leucopus',
                   'Peromyscus_maniculatus_bairdii', 'Onychomys_torridus', 'Myodes_glareolus', 'Microtus_ochrogaster',
                   'Phodopus_roborovskii', 'Cricetulus_griseus']
    res_tree, leaf_names = clean_nodes(newick_str, keep_leafs)
    assert len(leaf_names) == 50


def test_remove_some_leaves():
    res_tree, leaf_names = remove_some_leaves('100507096_no_supports.tree', 50)
    assert len(leaf_names) == 50
