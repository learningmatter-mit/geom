"""
Functions for comparing geometries and energies calculated at different levels of
theory.
"""

import os
import numpy as np
from matplotlib import rcParams
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm
from scipy.stats import spearmanr


from ase.build.rotate import minimize_rotation_and_translation as align
from ase import Atoms

try:
    from nff.utils.geom import compute_distances
    from nff.data import Dataset
except ImportError:
    print(("Could not import NFF utils for efficiently computing distances between "
           "conformers. Please install NFF from https://github.com/learningmatter-mit"
           "/NeuralForceField and put it in your path."))

TRANSLATION = {"opt": "relative to seed CREST conf",
               "closest": "relative to closest CREST conf"}
rcParams.update({"font.size": 20})


def load_pickle(direc):
    """
    Load pickle files for each species
    """

    files = os.listdir(direc)

    overall_dict = {}
    for file in tqdm(files):
        path = os.path.join(direc, file)

        with open(path, 'rb') as f:
            dic = pickle.load(f)

        if 'conformers' in dic:
            smiles = dic['smiles']
            conf_dics = [sub_dic for sub_dic in dic['conformers']
                         if 'boltzmannweight' in sub_dic]
        else:
            conf_dics = list(dic.values())
            smiles = conf_dics[0]['smiles']

        for i, conf_dic in enumerate(conf_dics):
            if 'rd_mol' in conf_dic:
                rd_mol = conf_dic['rd_mol']
                conf = rd_mol.GetConformers()[0]
                pos = conf.GetPositions()
                nums = [i.GetAtomicNum() for i in rd_mol.GetAtoms()]
                conf_dic['xyz'] = pos
                conf_dic['nums'] = nums
                conf_dic.pop('rd_mol')
            conf_dics[i] = conf_dic

        overall_dict[smiles] = conf_dics

    return overall_dict


def align_censo_crest(censo_dict,
                      crest_dict,
                      smiles):
    """
    For each CENSO-optimized conformer, find the conformer from which it was seeed
    with CREST.
    """

    dics = [censo_dict, crest_dict]
    if any([smiles not in dic for dic in dics]):
        return

    censo_confs = censo_dict[smiles]
    crest_confs = crest_dict[smiles]

    compare_list = []
    censo_to_crest_idx = []

    for censo_conf_dic in censo_confs:
        confnum = censo_conf_dic['confnum']
        try:
            crest_idx = [i for i, dic in enumerate(crest_confs)
                         if dic['confnum'] == confnum][0]
        except Exception as e:
            print("Failed for smiles %s with error '%s'" % (smiles, e))
            break
        this_crest_dic = crest_confs[crest_idx]

        compare = {"xtb": {"xyz": this_crest_dic["xyz"],
                           "nums": this_crest_dic["nums"]},
                   "censo": {"xyz": censo_conf_dic["xyz"],
                             "nums": censo_conf_dic["nums"]}}

        compare_list.append(compare)
        censo_to_crest_idx.append(crest_idx)

    return compare_list, censo_to_crest_idx


def dics_to_dset(dics):

    nxyz_list = []

    for dic in dics:
        xyz = np.array(dic['xyz'])
        nums = np.array(dic['nums']).reshape(-1, 1)

        nxyz = np.concatenate([nums, xyz], axis=-1)
        nxyz_list.append(nxyz)

    props = {"nxyz": nxyz_list}
    dset = Dataset(props=props)

    return dset


def compare_closest(censo_dict,
                    crest_dict,
                    smiles):

    censo_dset = dics_to_dset(censo_dict[smiles])
    crest_dset = dics_to_dset(crest_dict[smiles])

    distance_mat, _ = compute_distances(dataset=censo_dset,
                                        device='cpu',
                                        dataset_1=crest_dset)

    closest_idx = distance_mat.argmin(dim=-1).tolist()
    rmsds = distance_mat.min(dim=-1).values.tolist()

    return rmsds, closest_idx


def compare_all_closest(censo_dict,
                        crest_dict):
    """
    Figure out how different a censo geometry is from its closest counterpart
    in crest.
    """

    common_smiles = [key for key in censo_dict.keys()
                     if key in crest_dict.keys()]

    rmsds = []
    closest_idx_dic = {}

    for smiles in tqdm(common_smiles):
        these_rmsds, closest_idx = compare_closest(censo_dict,
                                                   crest_dict,
                                                   smiles)
        rmsds += these_rmsds
        closest_idx_dic[smiles] = closest_idx

    mean = np.mean(rmsds)
    std = np.std(rmsds)

    return (rmsds, mean, std, closest_idx_dic)


def get_rmsd(target, atoms):
    align(target=target,
          atoms=atoms)

    pos_target = target.get_positions()
    pos_atoms = atoms.get_positions()
    rmsd = np.mean((pos_target - pos_atoms) ** 2) ** 0.5
    return rmsd


def get_rmsds(compare_list):
    rmsds = []
    for compare in compare_list:
        atoms_list = []
        for key in ['xtb', 'censo']:
            sub_dic = compare[key]
            nums = sub_dic['nums']
            xyz = sub_dic['xyz']
            atoms = Atoms(numbers=nums,
                          positions=xyz)
            atoms_list.append(atoms)

        rmsd = get_rmsd(target=atoms_list[0],
                        atoms=atoms_list[1])

        rmsds.append(rmsd)
    return rmsds


def get_all_opt_rmsds(censo_dict,
                      crest_dict):

    common_smiles = [key for key in censo_dict.keys()
                     if key in crest_dict.keys()]

    rmsds = []
    crest_idx_dic = {}

    for smiles in common_smiles:
        compare_list, crest_idx = align_censo_crest(censo_dict,
                                                    crest_dict,
                                                    smiles)
        rmsds += get_rmsds(compare_list)
        crest_idx_dic[smiles] = crest_idx

    mean = np.mean(rmsds)
    std = np.std(rmsds)

    return rmsds, mean, std, crest_idx_dic


def get_all_distances(crest_dict,
                      censo_dict):

    keys = ["rmsds", "mean", "std"]

    output_dic = {}
    out = get_all_opt_rmsds(censo_dict=censo_dict,
                            crest_dict=crest_dict)
    censo_to_seed_crest = out[-1]
    output_dic["opt"] = {key: val for key, val
                         in zip(keys, out[:3])}

    out = compare_all_closest(censo_dict=censo_dict,
                              crest_dict=crest_dict)
    censo_to_closest_crest = out[-1]

    output_dic["closest"] = {key: val for key, val
                             in zip(keys, out[:3])}

    out = (output_dic, censo_to_seed_crest, censo_to_closest_crest)

    return out


def plot_geometry_changes(output_dic):

    for key, sub_dic in output_dic.items():
        sim_type = TRANSLATION[key]
        title = "CENSO vs. CREST geometries,\n %s" % sim_type

        fig, ax = plt.subplots()
        plt.hist(sub_dic['rmsds'])
        plt.text(0.52, 0.75, "RMSD = %.2f $\\pm$\n %.2f $\\AA$" % (sub_dic['mean'],
                                                                sub_dic['std']),
                 transform=ax.transAxes)
        plt.xlabel(r"RMSD ($\AA$)")
        plt.ylabel("Count")
        plt.title(title, fontsize=18)
        plt.show()


def get_en_changes(censo_dict,
                   crest_dict,
                   censo_to_seed_crest,
                   censo_to_closest_crest):
    """
    Get changes in energetic ordering between crest and censo
    """

    idx_dics = {TRANSLATION["opt"]: censo_to_seed_crest,
                TRANSLATION["closest"]: censo_to_closest_crest}

    en_results = {}

    for name, idx_dic in idx_dics.items():
        en_results[name] = {}
        common_smiles = [key for key in censo_dict.keys()
                         if key in crest_dict.keys()]

        for smiles in common_smiles:
            censo_confs = censo_dict[smiles]
            all_crest_confs = crest_dict[smiles]
            crest_confs = [all_crest_confs[i] for i in idx_dic[smiles]]

            if len(crest_confs) != len(censo_confs):
                continue

            crest_ens = np.array([i['relativeenergy'] for i in crest_confs])

            censo_ens = np.array([i['totalenergy'] for i in censo_confs])
            censo_ens -= np.min(censo_ens)
            censo_ens *= 672.5

            spear = spearmanr(crest_ens,
                              censo_ens)

            en_results[name][smiles] = {"crest": crest_ens,
                                        "censo": censo_ens,
                                        "spearman": spear.correlation}

    return en_results


def plot_en_changes(en_results):

    for sim_type, dic in en_results.items():
        spearman = np.array([val['spearman'] for val in dic.values()])
        spearman = spearman[np.isfinite(spearman)]

        mean = np.mean(spearman)
        std = np.std(spearman)

        title = "CREST vs. CENSO energetic ordering,\n %s" % sim_type

        fig, ax = plt.subplots()
        plt.hist(spearman)
        plt.title(title, fontsize=18)
        plt.xlabel(r"Spearman $\rho$")
        plt.ylabel("Count")
        plt.text(0.1, 0.87, r"$\rho = %.2f +/- %.2f$" % (
            mean, std),
            transform=ax.transAxes)
        plt.show()
