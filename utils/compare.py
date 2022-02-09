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
from sklearn.metrics import r2_score

from ase.build.rotate import minimize_rotation_and_translation as align
from ase import Atoms


try:
    from nff.utils.geom import compute_distances
    from nff.data import Dataset
    from nff.utils.constants import AU_TO_KCAL
except ImportError as e:
    print(e)
    print(("Could not import NFF utils for efficiently computing distances between "
           "conformers. Please check the error message, and make sure you have NFF "
           "installed and in your path. You can download it at https://github.com"
           "/learningmatter-mit/NeuralForceField and put it in your path."))

TRANSLATION = {"opt": "relative to seed CREST conf",
               "closest": "relative to closest CREST conf"}
rcParams.update({"font.size": 20})

DPI = 300


def load_pickle(direc, max_num=None):
    """
    Load pickle files for each species
    """

    files = os.listdir(direc)

    overall_dict = {}

    for file in tqdm(files):
        path = os.path.join(direc, file)

        try:
            with open(path, 'rb') as f:
                dic = pickle.load(f)
        except Exception as e:
            print(e)
            continue

        if max_num is not None:
            if len(overall_dict) > max_num:
                break

        if 'conformers' in dic:
            smiles = dic['smiles']
            conf_dics = dic['conformers']
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

        # some conformers cannot be reliably traced
        # back to a CREST conformer, because they were
        # incorrectly distorted by Orca 5.0.1 during the
        # optimization. These have to stay in the ensemble
        # because other conformers with the same structure
        # may have been removed as duplicates, so we had to leave them.

        if not censo_conf_dic.get('trust_confnum', True):
            continue

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

    num_atoms = len(pos_target)
    rmsd = np.sum((pos_target - pos_atoms) ** 2 / num_atoms) ** 0.5
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


def get_save_path(save_dir,
                  save_name,
                  key):

    if key:
        split = save_name.split(".")
        new_name = "%s_%s.%s" % (split[0], key, split[1])
    else:
        new_name = save_name
    save_path = os.path.join(save_dir, new_name)

    return save_path


def plot_geometry_changes(output_dic,
                          save_dir=None,
                          save_name=None):

    for key, sub_dic in output_dic.items():
        sim_type = TRANSLATION[key]
        title = "CENSO vs. CREST geometries,\n %s" % sim_type

        print(title)

        fig, ax = plt.subplots()
        plt.hist(sub_dic['rmsds'])
        plt.text(0.55, 0.7, "RMSD = %.2f $\\pm$\n %.2f $\\AA$" % (sub_dic['mean'],
                                                                  sub_dic['std']),
                 transform=ax.transAxes,
                 fontsize=16)
        plt.xlabel(r"RMSD ($\AA$)")
        plt.ylabel("Count")
        # plt.title(title, fontsize=16)
        [i.set_linewidth(2) for i in ax.spines.values()]
        plt.tight_layout()

        if all([save_dir is not None, save_name is not None]):
            save_path = get_save_path(save_dir=save_dir,
                                      save_name=save_name,
                                      key=key)
            plt.savefig(save_path,
                        dpi=DPI)
        plt.show()


def get_pct_contained(censo_confs,
                      all_crest_confs,
                      idx):

    crest_correct = [all_crest_confs[i] for i in idx]
    contained_crest_ens = np.array([i['relativeenergy'] for
                                    i in crest_correct])

    max_crest_en = np.max(contained_crest_ens)
    if max_crest_en == 0:
        pct_correct_contained = 100
        rel_increase = 0
        return pct_correct_contained, rel_increase

    crest_contained = [i for i in all_crest_confs if
                       i['relativeenergy'] <= max_crest_en]
    pct_correct_contained = len(crest_correct) / len(crest_contained) * 100

    rel_increase = (len(crest_contained) - len(idx)) / len(idx) * 100

    return pct_correct_contained, rel_increase


def get_crest_censo_en_changes(censo_dict,
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

            crest_ens = np.array([i['relativeenergy']
                                  for i in crest_confs])

            censo_ens = np.array([i['totalenergy'] for i in censo_confs])
            censo_ens -= np.min(censo_ens)
            censo_ens *= 627.5

            spear = spearmanr(crest_ens,
                              censo_ens)

            # now look at the entire ensemble of each to see how much of censo
            # is contained within the top X% of crest

            out = get_pct_contained(censo_confs=censo_confs,
                                    all_crest_confs=all_crest_confs,
                                    idx=idx_dic[smiles])
            pct_contained, rel_increase = out

            en_results[name][smiles] = {"crest": crest_ens,
                                        "censo": censo_ens,
                                        "spearman": spear.correlation,
                                        "pct_contained": pct_contained,
                                        "rel_increase": rel_increase}

    return en_results


def get_censo_sp_en_changes(censo_dict,
                            crest_dict,
                            censo_to_seed_crest,
                            censo_to_closest_crest,
                            sp_key):
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

            crest_confs = crest_dict[smiles]

            crest_idx = idx_dic[smiles]
            censo_idx = [i for i, idx in enumerate(crest_idx) if sp_key
                         in crest_confs[idx]]
            sp_idx = [idx_dic[smiles][i] for i in censo_idx]

            censo_confs = [censo_dict[smiles][i] for i in censo_idx]
            sp_confs = [crest_confs[i] for i in sp_idx]

            sp_ens = np.array([i[sp_key]['totalenergy']
                               for i in sp_confs]) * 627.5
            censo_ens = np.array([i['totalenergy']
                                  for i in censo_confs]) * 627.5

            if sp_ens.shape[0] == 0:
                spear = np.nan
            else:
                spear = spearmanr(sp_ens,
                                  censo_ens).correlation

            en_results[name][smiles] = {"sp": sp_ens,
                                        "censo": censo_ens,
                                        "spearman": spear}

    return en_results


def get_crest_sp_en_changes(crest_dict,
                            sp_key):
    """
    Get changes in energetic ordering between crest and censo
    """

    en_results = {}
    for smiles, confs in crest_dict.items():
        # import pdb
        # pdb.set_trace()
        confs_w_sp = [i for i in confs if sp_key in i]

        crest_ens = np.array([i['totalenergy'] for i in confs_w_sp])
        sp_ens = np.array([i[sp_key]['totalenergy'] for i in confs_w_sp])

        if sp_ens.shape[0] == 0 or crest_ens.shape[0] == 0:
            continue

        for these_ens in [crest_ens, sp_ens]:
            these_ens -= np.min(these_ens)
            these_ens *= 627.5

        spear = spearmanr(crest_ens,
                          sp_ens)
        frac_w_sp = len(confs_w_sp) / len(confs)
        en_results[smiles] = {"crest": crest_ens,
                              "sp": sp_ens,
                              "spearman": spear.correlation,
                              "frac_w_sp": frac_w_sp}

    return en_results


def plot_en_changes(en_results,
                    save_dir=None,
                    save_name=None):

    for sim_type, dic in en_results.items():
        spearman = np.array([val['spearman'] for val in dic.values()])
        spearman = spearman[np.isfinite(spearman)]

        mean = np.mean(spearman)
        std = np.std(spearman)

        # title = "CREST vs. CENSO energetic ordering,\n %s" % sim_type

        fig, ax = plt.subplots()
        plt.hist(spearman)
        # plt.title(title, fontsize=18)
        plt.xlabel(r"CREST / CENSO Spearman $\rho$")
        plt.ylabel("Count")
        plt.text(0.03, 0.8, r"$\rho = %.2f \pm %.2f$" % (
            mean, std),
            transform=ax.transAxes,
            fontsize=16)
        [i.set_linewidth(2) for i in ax.spines.values()]
        plt.tight_layout()

        if all([save_dir is not None, save_name is not None]):
            save_path = get_save_path(save_dir=save_dir,
                                      save_name=save_name,
                                      key=sim_type)
            plt.savefig(save_path, dpi=DPI)

        plt.show()


def plot_crest_sp_ens(sp_en_results,
                      save_dir=None,
                      save_name=None,
                      include_threshold=0.99):

    spearman = np.array([val['spearman'] for val in sp_en_results.values()
                         if val['frac_w_sp'] >= include_threshold])
    spearman = spearman[np.isfinite(spearman)]

    mean = np.mean(spearman)
    std = np.std(spearman)

    title = "CREST vs. single point energetic ordering"

    fig, ax = plt.subplots()
    plt.hist(spearman)
    plt.title(title, fontsize=18)
    plt.xlabel(r"Spearman $\rho$")
    plt.ylabel("Count")
    plt.text(0.03, 0.8, r"$\rho = %.2f \pm %.2f$" % (
        mean, std),
        transform=ax.transAxes,
        fontsize=16)
    [i.set_linewidth(2) for i in ax.spines.values()]
    plt.tight_layout()

    if all([save_dir is not None, save_name is not None]):
        save_path = get_save_path(save_dir=save_dir,
                                  save_name=save_name,
                                  key=None)
        plt.savefig(save_path, dpi=DPI)

    plt.show()


def plot_pct_contained(en_results,
                       save_dir=None,
                       save_name=None):

    for sim_type, dic in en_results.items():
        pct_contained = np.array([val['pct_contained']
                                  for val in dic.values()])
        pct_contained = pct_contained[np.isfinite(pct_contained)]

        mean = np.mean(pct_contained)
        std = np.std(pct_contained)

        title = "CREST vs. CENSO energetic ordering,\n %s" % sim_type

        fig, ax = plt.subplots()
        plt.hist(pct_contained)
        plt.title(title, fontsize=18)
        plt.xlabel("Conformers correct (%)")
        plt.ylabel("Count")
        plt.text(0.2, 0.8, r"Correct $= %d \pm %d$%%" % (
            mean, std),
            transform=ax.transAxes,
            fontsize=16)
        [i.set_linewidth(2) for i in ax.spines.values()]
        plt.tight_layout()

        if all([save_dir is not None, save_name is not None]):
            save_path = get_save_path(save_dir=save_dir,
                                      save_name=save_name,
                                      key=sim_type)
            plt.savefig(save_path, dpi=DPI)

        plt.show()


def plot_rel_increase(en_results,
                      save_dir=None,
                      save_name=None):

    for sim_type, dic in en_results.items():
        rel_increase = np.array([val['rel_increase']
                                 for val in dic.values()])
        rel_increase = rel_increase[np.isfinite(rel_increase)]

        mean = np.mean(rel_increase)
        std = np.std(rel_increase)

        title = "CREST vs. CENSO energetic ordering,\n %s" % sim_type

        fig, ax = plt.subplots()
        plt.hist(rel_increase)
        plt.title(title, fontsize=18)
        plt.xlabel("Z score (%)")
        plt.ylabel("Count")
        plt.text(0.2, 0.8, r"Correct $= %d \pm %d$%%" % (
            mean, std),
            transform=ax.transAxes,
            fontsize=16)
        [i.set_linewidth(2) for i in ax.spines.values()]
        plt.tight_layout()

        if all([save_dir is not None, save_name is not None]):
            save_path = get_save_path(save_dir=save_dir,
                                      save_name=save_name,
                                      key=sim_type)
            plt.savefig(save_path, dpi=DPI)

        plt.show()


def dft_ens_for_comparison(crest_dict,
                           censo_dict,
                           censo_to_closest_crest,
                           dft_name,
                           method):

    common_smiles = [key for key in crest_dict.keys()
                     if key in censo_dict.keys()]

    opt_dft_ens = {}
    sp_dft_ens = {}
    opt_dft_free_ens = {}

    for smiles in tqdm(common_smiles):

        censo_confs = censo_dict[smiles]
        crest_confs = crest_dict[smiles]

        opt_dft_ens[smiles] = []
        opt_dft_free_ens[smiles] = []
        sp_dft_ens[smiles] = []

        missing_match = False

        for i, censo_conf in enumerate(censo_confs):

            if method == 'opt':
                confnum = censo_conf['confnum']
                if not censo_conf.get('trust_confnum', True):
                    continue

                matching_crest_confs = [conf for conf in crest_confs if
                                        conf.get('confnum') == confnum]

                if len(matching_crest_confs) != 1:
                    missing_match = True
                    print(("Conformer match missing for smiles %s "
                           "and conformer %d" % (smiles, confnum)))
                    break
                crest_conf = matching_crest_confs[0]

            else:
                idx = censo_to_closest_crest[smiles][i]
                crest_conf = crest_confs[idx]

            if missing_match:
                continue

            if dft_name not in crest_conf:
                continue

            opt_dft_ens[smiles].append(censo_conf['totalenergy'])
            opt_dft_free_ens[smiles].append(censo_conf['deltaGtot'])
            sp_dft_ens[smiles].append(crest_conf[dft_name]['totalenergy'])

        if missing_match:
            for dic in [opt_dft_ens, opt_dft_free_ens, sp_dft_ens]:
                dic.pop(smiles)
            continue

        for dic in [opt_dft_ens, opt_dft_free_ens, sp_dft_ens]:
            ens = np.array(dic[smiles])
            if len(ens) == 0:
                dic.pop(smiles)
                continue

            ens -= np.min(ens)
            ens *= AU_TO_KCAL['energy']
            dic[smiles] = ens

    return opt_dft_ens, opt_dft_free_ens, sp_dft_ens


def get_spearmans(dic,
                  other_dic):
    rhos = []

    for key, these_ens in dic.items():
        other_ens = other_dic[key]
        assert len(these_ens) == len(other_ens)

        spear = spearmanr(these_ens, other_ens)
        rho = spear.correlation
        rhos.append(rho)

    return rhos


def get_and_plot_rho(dic,
                     other_dic,
                     title,
                     key,
                     save_dir=None,
                     save_name=None,
                     spear_name=None):

    rhos = get_spearmans(dic=dic,
                         other_dic=other_dic)

    rhos = np.array(rhos)
    rhos = rhos[np.isfinite(rhos)]

    mean = np.mean(rhos)
    std = np.std(rhos)

    if spear_name is not None:
        xlabel = r"%s Spearman $\rho$" % spear_name
    else:
        xlabel = r"Spearman $\rho$"

    fig, ax = plt.subplots()
    plt.hist(rhos)
    if title is not None:
        plt.title(title, fontsize=18)
    plt.text(0.03, 0.8, r"$\rho = %.2f \pm %.2f$" % (mean,
                                                     std),
             transform=ax.transAxes,
             fontsize=16)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    [i.set_linewidth(2) for i in ax.spines.values()]
    plt.tight_layout()

    if all([save_dir is not None, save_name is not None]):
        save_path = get_save_path(save_dir=save_dir,
                                  save_name=save_name,
                                  key=key)
        plt.savefig(save_path, dpi=DPI)
    plt.show()


def get_rel_en(sub_dics, key):
    ens = [i[key] for i in sub_dics]
    ens = np.array(ens)
    ens -= np.min(ens)
    ens *= AU_TO_KCAL['energy']

    return ens


def plot_energy_comparison(crest_dict,
                           censo_dict,
                           censo_to_closest_crest,
                           dft_name,
                           save_dir=None,
                           save_name=None):

    methods = ['opt', 'closest']
    for method in methods:
        out = dft_ens_for_comparison(crest_dict=crest_dict,
                                     censo_dict=censo_dict,
                                     censo_to_closest_crest=censo_to_closest_crest,
                                     dft_name=dft_name,
                                     method=method)

        opt_dft_ens, opt_dft_free_ens, sp_dft_ens = out

        get_and_plot_rho(dic=opt_dft_ens,
                         other_dic=sp_dft_ens,
                         title=('DFT single point vs. opt energetic\n'
                                'ordering, %s') % (TRANSLATION[method]),
                         save_dir=save_dir,
                         save_name=save_name,
                         key=method)


def plot_free_en_comparison(censo_dict,
                            save_dir=None,
                            save_name=None):

    censo_ens = {smiles: get_rel_en(sub_dics, 'totalenergy')
                 for smiles, sub_dics in censo_dict.items()}
    censo_free_ens = {smiles: get_rel_en(sub_dics, 'deltaGtot')
                      for smiles, sub_dics in censo_dict.items()}

    get_and_plot_rho(dic=censo_ens,
                     other_dic=censo_free_ens,
                     # title='DFT energy vs. free energy',
                     title=None,
                     save_dir=save_dir,
                     save_name=save_name,
                     key=None,
                     spear_name='G/E')


def get_sp_and_crest_ens(crest_dict,
                         cutoff,
                         sp_key):

    rel_crest_ens = []
    rel_r2scan_ens = []
    rho_scores = []

    for smiles, confs in crest_dict.items():
        confs_w_dft = [conf for conf in confs if sp_key in conf]
        frac = len(confs_w_dft) / len(confs)
        if frac < cutoff:
            continue

        crest_ens = [conf['totalenergy'] for conf in confs_w_dft]
        r2scan_ens = [conf[sp_key]['totalenergy']
                      for conf in confs_w_dft]

        these_rel_crest = (np.array(crest_ens) - min(crest_ens)) * 627.5
        these_rel_r2scan = (np.array(r2scan_ens) - min(r2scan_ens)) * 627.5

        rel_crest_ens += these_rel_crest.tolist()
        rel_r2scan_ens += these_rel_r2scan.tolist()

        if len(these_rel_crest) != 1:
            corr = spearmanr(these_rel_crest, these_rel_r2scan).correlation
            rho_scores.append(corr)

    rel_crest_ens = np.array(rel_crest_ens)
    rel_r2scan_ens = np.array(rel_r2scan_ens)
    rho_scores = np.array(rho_scores)

    return rel_crest_ens, rel_r2scan_ens, rho_scores


def plot_conf_ens(rel_crest_ens,
                  rel_dft_ens,
                  save_path):

    ideal = np.linspace(min(rel_crest_ens), max(rel_crest_ens), 100)
    mae = abs(rel_crest_ens - rel_dft_ens).mean()
    rho = spearmanr(rel_crest_ens, rel_dft_ens).correlation

    fig, ax = plt.subplots()
    plt.hexbin(rel_crest_ens,
               rel_dft_ens,
               mincnt=1,
               gridsize=20)
    plt.plot(ideal, ideal, '--',
             linewidth=3,
             color='white')
    plt.xlabel("GFN2-xTB (kcal/mol)")
    plt.ylabel("r2scan-3c (kcal/mol)")
    plt.tight_layout()
    plt.text(0.035, 0.87, r"$\mathrm{MAE} = %.2f$ kcal/mol" % mae,
             transform=ax.transAxes,
             fontsize=16)
    plt.text(0.035, 0.77, r"$\rho = %.2f$" % rho,
             transform=ax.transAxes,
             fontsize=16)
    [i.set_linewidth(2) for i in ax.spines.values()]
    plt.savefig(save_path, dpi=DPI)
    plt.show()


def plot_conf_rhos(rho_scores,
                   save_path,
                   text_loc=None):

    mean = np.mean(rho_scores)
    std = np.std(rho_scores)

    text = r"$\rho = %.2f \pm %.2f$" % (mean, std)
    if text_loc is None:
        text_loc = [0.03, 0.8]

    fig, ax = plt.subplots()
    plt.text(text_loc[0],
             text_loc[1],
             text,
             transform=ax.transAxes,
             fontsize=16)
    plt.hist(rho_scores)
    plt.ylabel("Count")
    plt.xlabel(r"xTB / r2scan Spearman $\rho$")
    [i.set_linewidth(2) for i in ax.spines.values()]
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI)
    plt.show()


def plot_crest_vs_sp(crest_dict,
                     sp_key,
                     save_dir,
                     cutoffs=[0, 0.99],
                     text_locs=None):

    # save_dir = '/home/saxelrod/plots'
    if text_locs is None:
        text_locs = [None] * len(cutoffs)

    for i, cutoff in enumerate(cutoffs):

        print("Cutoff for ensemble completeness: %.2f" % cutoff)

        out = get_sp_and_crest_ens(crest_dict=crest_dict,
                                   cutoff=cutoff,
                                   sp_key=sp_key)
        rel_crest_ens, rel_dft_ens, rho_scores = out

        mae = abs(rel_crest_ens - rel_dft_ens).mean()
        r2 = r2_score(rel_crest_ens, rel_dft_ens)
        print("MAE: %.2f kcal/mol" % mae)
        print("R^2: %.2f" % r2)

        save_path = os.path.join(
            save_dir, 'mae_sp_cutoff_%d.png' % (int(cutoff * 100)))
        plot_conf_ens(rel_crest_ens=rel_crest_ens,
                      rel_dft_ens=rel_dft_ens,
                      save_path=save_path)

        save_path = os.path.join(save_dir,
                                 'rho_sp_cutoff_%d.png' % (int(cutoff * 100)))

        plot_conf_rhos(rho_scores=rho_scores,
                       save_path=save_path,
                       text_loc=text_locs[i])
