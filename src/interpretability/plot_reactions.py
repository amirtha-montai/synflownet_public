"""
Use graphviz to visualize the reactions from SynFlowNet.
Given the reaction trajectory of a synflownet, plot the reaction trajectories
Plot for molecules in https://www.notion.so/montai/Plotting-the-reaction-trajectories-23e39e26fde38043865aeeecc9093900?source=copy_link#23f39e26fde38057a5d7e6e78624da3a

"""

import re
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, rdChemReactions

TEMPLATES_FILE = "/home/ubuntu/synflownet/src/synflownet/data/templates/hb.txt"
BUILDING_BLOCKS_FILE = "/home/ubuntu/synflownet/src/synflownet/data/building_blocks/enamine_bbs.txt"


with open(TEMPLATES_FILE, "r") as f:
    all_reactions = f.readlines()
all_reactions = [x.strip() for x in all_reactions if x.strip()]


with open(BUILDING_BLOCKS_FILE, "r") as f:
    all_building_blocks = f.readlines()
all_building_blocks = [x.strip() for x in all_building_blocks if x.strip()]

RXN_START_INDEX_OFFSET = 1  # IF THE TRAJ HAS 34, IT MEANS ACCESS REACTION[35]
BB_START_INDEX_OFFSET = 0

def get_current_task_dir():
    return Path.cwd()

# For Smiles in the slide deck
SMILES_TRAJ_DICT = {
    "CCCN(C)CC1CCN(C(=O)CC(N)c2ccc(OC)nc2)C1":"[('', <GraphActionType.AddFirstReactant, 2119>), ('COc1ccc(C(N)CC(=O)O)cn1', <GraphActionType.ReactBi, 72, 2122>), ('CCCN(C)CC1CCN(C(=O)CC(N)c2ccc(OC)nc2)C1', <GraphActionType.ReactBi, 87, 508>), ('CCCN(C)CC1CCN(C(=O)CC(N)c2ccc(OC)nc2)C1', <GraphActionType.Stop, 0, 0>)]"
}

def process_action_types_1(traj, bb_start_index, rxn_start_index):
    """
    Process the first type of action types from the trajectory string.
    The type of actions include AddFirstReactant and ReactUni.
    They only take in BB or RXN, not both.

    traj : str
        The trajectory string containing actions and their parameters.
    bb_start_index: int
        The starting index for building blocks in the all_building_blocks list.
    rxn_start_index: int
        The starting index for reactions in the all_reactions list.
    """
    final_dict_results = {}
    tuple_pattern_addfirstreactant = r"\('([^']*)', <GraphActionType\.([^,]+), (\d+)>\)"

    first_reactant_matches = []
    first_reactant_indices = []
    # TODO: Fix logic of index so it doesnt break
    for match in re.finditer(tuple_pattern_addfirstreactant, traj):
        smiles, action_type, id_ = match.groups()
        first_reactant_matches.append((smiles, action_type, id_))
        first_reactant_indices.append(match.start())

    if first_reactant_matches:
        for item, index in zip(first_reactant_matches, first_reactant_indices):
            action_type_str = item[1]
            if action_type_str == "AddFirstReactant":
                molecule = item[0]
                bb = int(item[2])
                bb_id = bb
                rxn_id = None
                bb = all_building_blocks[bb + bb_start_index] if bb < len(all_building_blocks) else None
                action_type = {
                    "action": "AddFirstReactant",
                    "bb": bb,
                    "rxn": None,
                    "input_mol": None if molecule == "" else molecule,
                    "bb_id": bb_id,
                    "rxn_id": rxn_id,
                }
            elif action_type_str == "ReactUni":
                molecule = item[0]
                rxn = int(item[2])
                bb_id = None
                rxn_id = int(item[2])
                rxn = all_reactions[rxn + rxn_start_index] if rxn < len(all_reactions) else None
                action_type = {
                    "action": "ReactUni",
                    "rxn": rxn,
                    "bb": None,
                    "input_mol": molecule,
                    "bb_id": bb_id,
                    "rxn_id": rxn_id,
                }
            else:
                raise ValueError(f"Unknown action type: {action_type_str} in {item}")
            final_dict_results[index] = action_type
    return final_dict_results


def process_action_types_2(traj, bb_start_index, rxn_start_index):
    """
    Process the second type of action types from the trajectory string.
    The type of actions include ReactBi, AddReactant, and Stop.
    They take in both BB and RXN.
    traj : str
        The trajectory string containing actions and their parameters.
    bb_start_index: int
        The starting index for building blocks in the all_building_blocks list.
    rxn_start_index: int
        The starting index for reactions in the all_reactions list.
    """
    final_dict_results = {}
    tuple_pattern = r"\('([^']*)', <GraphActionType\.([^,]+), (\d+), (\d+)>\)"

    first_reactant_matches = []
    first_reactant_indices = []
    # TODO: Fix logic of index so it doesnt break
    for match in re.finditer(tuple_pattern, traj):
        smiles, action_type, id_, id2_ = match.groups()
        first_reactant_matches.append((smiles, action_type, id_, id2_))
        first_reactant_indices.append(match.start())
    if first_reactant_matches:
        for item, index in zip(first_reactant_matches, first_reactant_indices):
            if index in final_dict_results:
                print(f"Index {index} already processed, skipping: {final_dict_results[index]}")
                continue  # Skip if already processed
            molecule = item[0]
            action_type_str = item[1]

            rxn = int(item[2])
            bb = int(item[3])
            rxn_id = int(item[2])
            bb_id = int(item[3])
            rxn = all_reactions[rxn + rxn_start_index] if rxn < len(all_reactions) else None
            bb = all_building_blocks[bb + bb_start_index] if bb < len(all_building_blocks) else None
            # stop has 0, 0 so we need to check
            if "Stop" in action_type_str:
                action_type = {
                    "action": "Stop",
                    "rxn": None,
                    "bb": None,
                    "input_mol": molecule,
                    "bb_id": bb_id,
                    "rxn_id": rxn_id,
                }
            elif "ReactBi" in action_type_str:
                action_type = {
                    "action": "ReactBi",
                    "rxn": rxn,
                    "bb": bb,
                    "input_mol": molecule,
                    "bb_id": bb_id,
                    "rxn_id": rxn_id,
                }
            elif "AddReactant" in action_type_str:
                print(f"AddReactant found in matches, but should it be here: {item}")
                action_type = {
                    "action": "AddReactant",
                    "rxn": None,
                    "bb": bb,
                    "input_mol": molecule,
                    "bb_id": bb_id,
                    "rxn_id": rxn_id,
                }
            else:
                raise ValueError(f"Unknown action type: {action_type_str}")
            final_dict_results[index] = action_type
    return final_dict_results


def parse_graph_action_list_with_mapping(
    smi, traj, bb_start_index=BB_START_INDEX_OFFSET, rxn_start_index=RXN_START_INDEX_OFFSET
):
    """
    Parses a string representation of a list of GraphAction objects into a list of tuples.
    Each tuple contains a molecule string and a GraphAction object.

    # Note : ['AddFirstReactant', 'ReactUni'] are of type first reactant matches
    # Note:   ['ReactBi', 'AddReactant','Stop']  are of type other matches

    """
    results_action_types_1 = process_action_types_1(traj, bb_start_index, rxn_start_index)
    results_action_types_2 = process_action_types_2(traj, bb_start_index, rxn_start_index)

    final_dict_results = {**results_action_types_1, **results_action_types_2}  # Merge the two dictionaries
    # Sort the final results by index to maintain the original order
    sorted_results = [final_dict_results[i] for i in sorted(final_dict_results.keys())]
    final_dict_results.clear()  # Clear the dictionary to free memory

    assert sorted_results[-1]["input_mol"] == smi, (
        f"Last molecule in sorted results does not match input SMILES: {sorted_results[-1]['input_mol']} != {smi}"
    )
    print(f"Parsed {len(sorted_results)} actions from trajectory for SMILES: {smi}")

    return sorted_results


def plot_reaction_items_BI(reactant_1_smiles, reactant_2_smiles, product_of_bi, step_num, inchikey):
    reactant1 = Chem.MolFromSmiles(reactant_1_smiles)
    reactant2 = Chem.MolFromSmiles(reactant_2_smiles)
    product = Chem.MolFromSmiles(product_of_bi)

    # Build the reaction object
    rxn = rdChemReactions.ChemicalReaction()
    rxn.AddReactantTemplate(reactant1)
    rxn.AddReactantTemplate(reactant2)
    rxn.AddProductTemplate(product)

    # Optional: Set names for labeling
    reactant1.SetProp("_Name", "Reactant 1")
    reactant2.SetProp("_Name", "Reactant 2")
    product.SetProp("_Name", "Product")

    rxn_plot_img = Draw.ReactionToImage(rxn, subImgSize=(300, 300), useSVG=False)
    save_path = get_current_task_dir() / f"{inchikey}_reaction_bi_step_{step_num + 1}.png"
    rxn_plot_img.save(save_path)
    return rxn


def plot_reaction_items_UNI(reactant_1_smiles, product_of_UNI, step_num, inchikey):
    reactant1 = Chem.MolFromSmiles(reactant_1_smiles)
    product = Chem.MolFromSmiles(product_of_UNI)

    # Build the reaction object
    rxn = rdChemReactions.ChemicalReaction()
    rxn.AddReactantTemplate(reactant1)
    rxn.AddProductTemplate(product)

    # Optional: Set names for labeling
    reactant1.SetProp("_Name", "Reactant 1")
    product.SetProp("_Name", "Product")

    rxn_plot_img = Draw.ReactionToImage(rxn, subImgSize=(300, 300), useSVG=False)
    save_path = get_current_task_dir() / f"{inchikey}_reaction_uni_step_{step_num + 1}.png"
    rxn_plot_img.save(save_path)
    return rxn


def plot_the_reaction(rxn_smarts, step_num, inchikey):
    rxn_plot = rdChemReactions.ReactionFromSmarts(rxn_smarts)
    actual_rxn_img = Draw.ReactionToImage(rxn_plot, subImgSize=(300, 300), useSVG=False)
    actual_save_path = get_current_task_dir() / f"{inchikey}_actual_reaction_{step_num + 1}.png"
    actual_rxn_img.save(actual_save_path)
    return rxn_plot


def plot_the_trajectory_of_synflowner_res(smile, traj, smile_id=""):
    """
    Plot the reaction trajectory of a SynFlowNet reaction.
    SMI : str
        The SMILES string of the final product.
    TRAJS : str
        The trajectory string containing actions and their parameters.
    """
    try:
        inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smile))
    except Exception as e:
        print(f"Error generating InChIKey for SMILES {smile}: {e}")
        inchikey = "invalid_inchikey_"
    inchikey = inchikey + smile_id
    sorted_trajectory = parse_graph_action_list_with_mapping(smile, traj)
    # save sorted trajectory to csv
    with open(get_current_task_dir() / f"{inchikey}_trajectory.csv", "w") as f:
        f.write("step_num,action,rxn,bb,input_mol,bb_id,rxn_id\n")
        for i, step in enumerate(sorted_trajectory):
            f.write(
                f"{i},{step['action']},{step['rxn']},{step['bb']},{step['input_mol']},{step['bb_id']},{step['rxn_id']}\n"
            )

    for i, step in enumerate(sorted_trajectory):
        print(step)
        if step["action"] == "ReactBi":
            reactant_1_smiles = step["bb"]
            reactant_2_smiles = step["input_mol"]
            step_temp = step
            if reactant_1_smiles is None or reactant_2_smiles is None:
                print(f"Skipping reaction {i + 1} due to missing reactants: {reactant_1_smiles}, {reactant_2_smiles}")
                # breakpoint()
            product_of_bi = sorted_trajectory[i + 1]["input_mol"]
            print(f"Plotting reaction {i + 1}: {reactant_1_smiles} + {reactant_2_smiles} -> {product_of_bi}")
            plot_reaction_items_BI(reactant_1_smiles, reactant_2_smiles, product_of_bi, step_num=i, inchikey=inchikey)
            plot_the_reaction(rxn_smarts=step["rxn"], step_num=i, inchikey=inchikey)

        elif step["action"] == "ReactUni":
            reactant_1_smiles = step["input_mol"]
            product_of_uni = sorted_trajectory[i + 1]["input_mol"]
            print(f"Plotting reaction {i + 1}: {reactant_1_smiles}  -> {product_of_uni}")
            plot_reaction_items_UNI(reactant_1_smiles, product_of_uni, step_num=i, inchikey=inchikey)
            plot_the_reaction(rxn_smarts=step["rxn"], step_num=i, inchikey=inchikey)


def main():

    i = 0
    for smile, traj in SMILES_TRAJ_DICT.items():
        print(f"Processing SMILES: {smile}")
        print(f"Trajectory: {traj}")
        plot_the_trajectory_of_synflowner_res(smile, traj, smile_id=f"_{i}")
        i += 1

if __name__ == "__main__":
    main()