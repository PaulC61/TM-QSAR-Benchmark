import pandas as pd
import numpy as np
import csv
import time
# from pyJoules.energy_meter import measure_energy
import statistics
from joblib import Parallel, delayed

# import polaris as po

# Data cleaning
from chembl_structure_pipeline.standardizer import standardize_mol as chembl_standardizer

# Descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs, rdFingerprintGenerator

# BINARIZER
from tm_binarizer import Binarizer as QuantileBinarizer

# Split functions and MoleculeDatasets
import useful_rdkit_utils as uru
from useful_rdkit_utils import GroupKFoldShuffle

# Models
# RF
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyRegressor

# TM
from tmu.models.regression.vanilla_regressor import TMRegressor
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine

# XGBoost
from xgboost import XGBClassifier, XGBRegressor

# HP tuning
import optuna
import os
os.environ['OMP_NUM_THREADS'] = '180'

#### ---- ARGUMENTS --- ####

# DATA CHARACTERIZATION
DATASET_SUBSET = [
    # if achieving high performance  on opioids - explore regression tasks
    "opioids/MOR_cutoff6.csv",
    "opioids/DOR_cutoff6.csv",
    "opioids/KOR_cutoff6.csv",
    "opioids/CYP2D6_cutoff6.csv",
    "opioids/CYP3A4_cutoff6.csv"
]

LEARNING_TASK = [
    "class",
    "class",
    "class",
    "class",
    "class"
    
]

SMILES_COL = [
    "SMILES",
    "SMILES",
    "SMILES",
    "SMILES",
    "SMILES"
]

PROP_COL = [
    "label",
    "label",
    "label",
    "label",
    "label"
]

GROUP_LST = [
    ("random", uru.get_random_clusters),
    # ("butina", uru.get_butina_clusters),
    ("scaffold", uru.get_bemis_murcko_clusters)
]

# DESCRIPTORS
# label, binary
DESCRIPTOR_SET = [
    "ECFP",
    "RDKit2D"
]
FP_SIZE = 2048
FP_RAD = 2

# MODELS
MODEL_LABELS = [
   "TsetlinMachine",
    "RandomForest",
    "XGBoost"
]

N_TM_EPOCHS = 50

# MODEL TRAINING
C_FACTOR = 16
N_TREES = 100
N_CLAUSES = N_TREES*C_FACTOR

# start, end, log, int
PARAM_GRIDS = [
    {
   "T": (1*N_CLAUSES, 10*N_CLAUSES, False, True), # int
   "s": (1, 7, False, False)
    },
    {
    "max_depth": (10, 100, False, True), # int
    "ccp_alpha": (0.001, 1.0, True, False)
    },
    {
    "max_depth": (5, 20, False, True), # int
    "learning_rate": (0.01, 1.0, True, False),
    "min_child_weight": (1, 10, True, False),
    "gamma": (0.01, 1.0, True, False),
    "subsample": (0.01, 1.0, True, False),
    'colsample_bytree': (0.01, 1.0, True, False),
    'colsample_bynode': (0.01, 1.0, True, False),
    'reg_alpha': (0.001, 1.0, True, False),
    'reg_lambda': (0.001, 1.0, True, False)
    }
]

N_TRIALS = 25

MACRO_OUT_FILENAME = "results/MACRO_TM_Benchmark_16_para"
MICRO_OUT_FILENAME = "results/MICRO_TM_Benchmark_16_para"


# Metrics in line with Deng et. al
# Classification Metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc, roc_curve, precision_recall_curve, accuracy_score, cohen_kappa_score, matthews_corrcoef, precision_score, recall_score

#AUPRC
def prc_auc_score(Y, Y_pred):
        precision, recall, _ = precision_recall_curve(Y, Y_pred)
        return auc(recall, precision)

# Precision_PPV, Precision_NPV
def ppv_npv_score(Y, Y_pred):
        fpr, tpr, proba = roc_curve(Y, Y_pred)
        optimal_proba_cutoff = sorted(list(zip(np.abs(tpr - fpr), proba)),\
                                      key=lambda i: i[0], reverse=True)[0][1]
        
        hard_Y_pred = [1 if p > optimal_proba_cutoff else 0 for p in Y_pred]

        return precision_score(Y, hard_Y_pred, pos_label=1), precision_score(Y, hard_Y_pred, pos_label=0)    


def write_clf_scores(Y, Y_pred, meta_info, model, dataset, writer):
    
    meta_info = meta_info + [model, dataset]
    
    roc_auc = roc_auc_score(Y, Y_pred, multi_class='ovr')
    roc_auc_line = meta_info + [roc_auc, "ROC_AUC"]
    writer.writerow(
        roc_auc_line
    )

    prc_auc = prc_auc_score(Y, Y_pred)
    prc_auc_line = meta_info + [prc_auc, "PRC_AUC"]
    writer.writerow(
        prc_auc_line
    )

    ppv, npv = ppv_npv_score(Y, Y_pred)
    ppv_line = meta_info + [ppv, "PPV"]
    writer.writerow(
        ppv_line
    )
    
    npv_line = meta_info + [npv, "NPV"]
    writer.writerow(
        npv_line
    )
    
    ## ACC
    #acc = accuracy_score(Y, Y_pred)
    #acc_line = meta_info + [acc, "ACC"]
    #writer.writerow(
    #    acc_line
    #)
    
    ## cohen_k
    #cohen_k = cohen_kappa_score(Y, Y_pred)
    #cohen_k_line = meta_info + [cohen_k, "Cohen_k"]
    #writer.writerow(
    #    cohen_k_line
    #)
    
    # MCC
    #mcc = matthews_corrcoef(Y, Y_pred)
    #mcc_line = meta_info + [mcc, "MCC"]
    #writer.writerow(
    #    mcc_line
    #)
    
    # precision
    #prec = precision_score(Y, Y_pred)
    #prec_line = meta_info + [prec, "Prec"]
    #writer.writerow(
    #    prec_line
    #)
    
    # recall
    #recall = recall_score(Y, Y_pred)
    #recall_line = meta_info + [recall, "Recall"]
    #writer.writerow(
    #    recall_line
    #)
    
    return None

def write_MICRO_clf_scores(Y_index, Y, Y_pred, meta_info, model, dataset, micro_writer):
    meta_info = meta_info + [model, dataset]
    for sample_indx in range(len(Y_index)):
        sample_pred = meta_info + [Y_index[sample_indx], Y[sample_indx], Y_pred[sample_indx]]
        micro_writer.writerow(
            sample_pred
        )
    return None


# Regression Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, matthews_corrcoef
from scipy.stats import pearsonr
from scipy.special import expit

def oos_r2_score(Y, Y_pred, Y_dummy_pred):
    mse_pred = mean_squared_error(Y, Y_pred)
    mse_dummy = mean_squared_error(Y, Y_dummy_pred)
    oos_r2 = 1 - (mse_pred/mse_dummy)
    return oos_r2

def write_reg_scores(Y, Y_pred, Y_dummy_pred, meta_info, model, dataset, writer):
    
    meta_info = meta_info + [model, dataset]
    
    rmse = root_mean_squared_error(Y, Y_pred)
    rmse_line = meta_info + [rmse, "RMSE"]
    writer.writerow(
        rmse_line
    )

    mae = mean_absolute_error(Y, Y_pred)
    mae_line = meta_info + [mae,"MAE"]
    writer.writerow(
        mae_line
    )

    pearson_r = pearsonr(Y, Y_pred)
    pearson_r_line = meta_info + [pearson_r, "Pearson_R"]
    writer.writerow(
        pearson_r_line
    )
    
    oos_r2 = oos_r2_score(Y, Y_pred, Y_dummy_pred)
    oos_r2_line = meta_info + [oos_r2,"R2"]
    writer.writerow(
        oos_r2_line
    )
    
    return None

# PARALLEL CLAUSE SUM
def cust_threshold(i):
    j = (i%2)*(-1)
    return 1 if (j > -1) else j

def parallel_tm_ccs(tm, X, n_classes=2, n_clauses=1000):
    active_clauses = tm.transform(X, inverted=False)
    weight_state = tm.get_state()
    mask = Parallel(n_jobs=22)(delayed(cust_threshold)(i) for i in np.arange(n_clauses))
    # mask = [1 if (i > -1) else i for i in ((np.arange(n_clauses)%2) *-1)]
    
    ccs = []
    for i_class in range(2):
        clause_class_weights = weight_state[i_class][0]*mask
        active_clause_class_weights = clause_class_weights*active_clauses[:, (i_class*n_clauses):(i_class+1)*n_clauses]
        class_clause_sum = active_clause_class_weights.sum(axis=1)
        ccs.append(class_clause_sum)
    return np.array(ccs)



# Generate clean molecules
def mol_from_smiles(smiles, *args, standardizer=chembl_standardizer) -> Chem.Mol:
    """
    Convert a SMILES string to an RDKit Mol object

    :param smiles: SMILES string. Required.
    :param standardizer: function to standardize the molecule. Defaults to ChEMBL standardizer. Use None to skip.
    :return: RDKit Mol object
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if not mol:
        return None
    if standardizer is None:
        return mol
    return standardizer(mol, check_exclusion=True, sanitize=True)

# Descriptor calculation
# ECFP fingerprints
def fp_to_np(fp):
    arr = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def gen_ecfp_arr(mol_df, mol_col, fp_size=1024, fp_radius=2, n_threads=-1):
    mols = list(mol_df[mol_col])
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_size)
    fps = fpg.GetFingerprints(mols, numThreads=n_threads)
    fps_np = np.array([fp_to_np(i) for i in fps], dtype=np.uint32)
    return fps_np

# RDKit 2D
def gen_rdkit2D_arr(mol_df, mol_col):
    mols = list(mol_df[mol_col])
    descrs = [Descriptors.CalcMolDescriptors(mol) for mol in mols]
    desc_np = pd.DataFrame(descrs).values
    return desc_np



def benchmark_CLF_objective(trial, X_train_in, X_val_in, Y_train_in, Y_val_in, model_label, param_grid):

    clf_params = dict()

    for param in param_grid:
        if param_grid[param][-1]:
            clf_params[param] = trial.suggest_int(param, param_grid[param][0], param_grid[param][1], log=param_grid[param][2])
        else:
            clf_params[param] = trial.suggest_float(param, param_grid[param][0], param_grid[param][1], log=param_grid[param][2])
    
    clf_model = None
    
    if model_label == "RandomForest":
        clf_params["n_estimators"] = N_TREES
        clf_model = RandomForestClassifier(**clf_params)
    elif model_label == "XGBoost":
        clf_params["n_estimators"] = N_TREES
        clf_model = XGBClassifier(**clf_params)
    else:
        clf_params["number_of_clauses"] = N_CLAUSES
        clf_params["weighted_clauses"] = True
        clf_model = MultiClassTsetlinMachine(**clf_params)

    # EPOCHS AND PRUNING
    if model_label != "TsetlinMachine":
        clf_model.fit(X_train_in, Y_train_in)
        Y_val_pred_prob = clf_model.predict_proba(X_val_in)[:, 1]
    else:
        for epoch in range(int(N_TM_EPOCHS*0.5)):
            clf_model.fit(X_train_in, Y_train_in, epochs=1, incremental=True)
            Y_val_ccs = parallel_tm_ccs(tm=clf_model, X=X_val_in, n_clauses=N_CLAUSES)
            Y_val_pred_prob = expit(Y_val_ccs/clf_model.T)[1]
        
            trial.report(roc_auc_score(y_true=Y_val_in, y_score=Y_val_pred_prob), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    del clf_model
    
    return roc_auc_score(y_true=Y_val_in, y_score=Y_val_pred_prob)
    
def benchmark_REG_objective(trial, X_train_in, X_val_in, Y_train_in, Y_val_in, model_label, param_grid):
    
    reg_params = dict()

    for param in param_grid:
        if param_grid[param][-1]:
            reg_params[param] = trial.suggest_int(param, param_grid[param][0], param_grid[param][1], log=param_grid[param][2])
        else:
            reg_params[param] = trial.suggest_float(param, param_grid[param][0], param_grid[param][1], log=param_grid[param][2])
    
    reg_model = None
    
    if model_label == "RandomForest":
        reg_params["n_estimators"] = N_TREES
        reg_model = RandomForestRegressor(**reg_params)
    elif model_label == "XGBoost":
        reg_params["n_estimators"] = N_TREES
        reg_model = XGBRegressor(**reg_params)
    else:
        reg_params["number_of_clauses"] = N_CLAUSES
        reg_params["weighted_clauses"] = True
        reg_model = TMRegressor(**reg_params)
    
    # EPOCHS AND PRUNING
    if model_label != "TsetlinMachine":
        reg_model.fit(X_train_in, Y_train_in)
        Y_val_pred = reg_model.predict(X_val_in)
    else:
        for epoch in range(int(N_TM_EPOCHS)):
            reg_model.fit(X_train_in, Y_train_in, epochs=1, incremental=True)
            Y_val_pred = reg_model.predict(X_val_in) 

            trial.report(root_mean_squared_error(y_true=Y_val_in, y_pred=Y_val_pred), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    return root_mean_squared_error(Y_val_in, Y_val_pred)


N_OUTER = 5
N_INNER = 5
K_VAL = 2
N_PARAM_SPLITS = 3
# 5 splits x 5 folds CV
# dataset N = 500 - 100,000
result_file = open(MACRO_OUT_FILENAME, 'a', newline='')

results_cols = ["TargetDataset", "Group", "Split", "Fold", "Descriptor","HPSearchTime", "TrainTime", "InferenceTime", "TotalTime", "Params", "Epochs", "Model", "Dataset", "Score", "ScoreType"]
explored_results = pd.read_csv(f"{MACRO_OUT_FILENAME}", names=results_cols)
config_cols = ["TargetDataset", "Group", "Split", "Fold", "Descriptor", "Model"]
explored_configs = explored_results.loc[:, config_cols]
explored_configs_arr = list(explored_configs.itertuples(index=False, name=None))

result_writer = csv.writer(result_file)

micro_result_file = open(MICRO_OUT_FILENAME, 'a', newline='')
micro_result_writer = csv.writer(micro_result_file)

for dataset_indx in range(len(DATASET_SUBSET)):
    task = LEARNING_TASK[dataset_indx]
    
    dataset_df = pd.read_csv(f"data/{DATASET_SUBSET[dataset_indx]}")
    dataset_df['mol'] = dataset_df[SMILES_COL[dataset_indx]].apply(lambda smi: mol_from_smiles(smi))
    dataset_df.dropna(subset=['mol'], inplace=True)
    
    for group_name, group_func in GROUP_LST:
        current_group = np.array(group_func(dataset_df[SMILES_COL[dataset_indx]]))
        for split in range(N_OUTER):
            kf_5 = GroupKFoldShuffle(n_splits=N_INNER, shuffle=True)
            print("Split", split)
            for fold, [train_indx, test_indx] in enumerate(kf_5.split(dataset_df, groups=current_group)):
                print("Fold", fold)
                
                train_df = dataset_df.loc[train_indx]
                test_df = dataset_df.loc[test_indx]
                
                Y_train, Y_test = np.array(train_df[PROP_COL[dataset_indx]]).flatten(), np.array(test_df[PROP_COL[dataset_indx]]).flatten()
                
                for descriptor_indx in range(len(DESCRIPTOR_SET)):
                    descriptor = DESCRIPTOR_SET[descriptor_indx]
                    
                    if descriptor == 'ECFP':
                        X_train, X_test = (
                            gen_ecfp_arr(mol_df=train_df, mol_col='mol', fp_size=FP_SIZE, fp_radius=FP_RAD, n_threads=-1),
                            gen_ecfp_arr(mol_df=test_df, mol_col='mol', fp_size=FP_SIZE, fp_radius=FP_RAD, n_threads=-1)
                        )
                    elif descriptor == 'RDKit2D':
                        X_train_cont, X_test_cont = (
                            gen_rdkit2D_arr(mol_df=train_df, mol_col='mol'),
                            gen_rdkit2D_arr(mol_df=test_df, mol_col='mol')
                        )

                        binarizer = QuantileBinarizer(resolution=10)
                        binarizer.fit(X_train_cont)

                        X_train, X_test = (
                            binarizer.transform(X_train_cont),
                            binarizer.transform(X_test_cont)
                        )
                        
                    
                    for model_indx in range(len(MODEL_LABELS)):
                        model_label = MODEL_LABELS[model_indx]
                        config_tuple = (DATASET_SUBSET[dataset_indx], group_name, split, fold, descriptor, model_label)
                        if config_tuple not in explored_configs_arr:
                            print(model_label)
                            param_grid = PARAM_GRIDS[model_indx]
                            
                            # PARAM_SEARCH
                            # two-fold CV does not need re-initializing for multiple different splits 1 x 2
                        
                            print("CV Param Search Started")
                            cv_best_params = []
                            hp_search_time = 0
                            kf_2 = GroupKFoldShuffle(n_splits=K_VAL, shuffle=False)
                            for v_fold, [train_val_indx, val_indx] in enumerate(kf_2.split(train_df, groups=current_group[train_indx])):
                                
                                train_val_df = train_df.iloc[train_val_indx]
                                val_df = train_df.iloc[val_indx]
                                
                                if descriptor == 'ECFP':
                                    X_train_val, X_val = (
                                        gen_ecfp_arr(mol_df=train_val_df, mol_col='mol', fp_size=FP_SIZE, fp_radius=FP_RAD, n_threads=-1),
                                        gen_ecfp_arr(mol_df=val_df, mol_col='mol', fp_size=FP_SIZE, fp_radius=FP_RAD, n_threads=-1)
                                    )
                                elif descriptor == 'RDKit2D':
                                    X_train_val_cont, X_val_cont = (
                                        gen_rdkit2D_arr(mol_df=train_val_df, mol_col='mol'),
                                        gen_rdkit2D_arr(mol_df=val_df, mol_col='mol')
                                    )

                                    binarizer = QuantileBinarizer(resolution=10)
                                    binarizer.fit(X_train_val_cont)

                                    X_train_val, X_val = (
                                        binarizer.transform(X_train_val_cont),
                                        binarizer.transform(X_val_cont)
                                    )
                                
                                Y_train_val, Y_val = (
                                    np.array(train_val_df[PROP_COL[dataset_indx]]).flatten(), 
                                    np.array(val_df[PROP_COL[dataset_indx]]).flatten()
                                )

                                study = optuna.create_study(
                                    direction = 'maximize',
                                    pruner = optuna.pruners.MedianPruner(
                                        n_startup_trials=5, n_warmup_steps=int(N_TM_EPOCHS*0.1)
                                    )
                                )

                                hp_search_start = time.time()
                                study.optimize(
                                    lambda trial: benchmark_CLF_objective(
                                        trial,
                                        X_train_in = X_train_val,
                                        X_val_in = X_val,
                                        Y_train_in = Y_train_val,
                                        Y_val_in = Y_val,
                                        model_label = model_label,
                                        param_grid = param_grid
                                    ),
                                    n_trials=N_TRIALS
                                )
                                hp_search_end = time.time()
                                hp_search_time = hp_search_time + (hp_search_end - hp_search_start)

                                cv_best_params.append(study.best_params)
                            
                            # start here for avg_best_params
                            
                            mean_best_params = {}
                            for param in param_grid:
                                isInt = param_grid[param][-1]
                                fold_values = []
                                for k_fold in range(K_VAL):
                                    fold_best_params = cv_best_params[k_fold]
                                    fold_values.append(fold_best_params[param])
                                if isInt:
                                    mean_best_params[param] = int(statistics.mean(fold_values))
                                else:
                                    mean_best_params[param] = statistics.mean(fold_values)
                            
                            final_params = mean_best_params
                            
                            if model_label != "TsetlinMachine":
                                final_params["n_estimators"] = N_TREES
                                if model_label == "RandomForest":
                                    clf_model = RandomForestClassifier(**final_params)
                                else:
                                    clf_model = XGBClassifier(**final_params)
                                
                                # BEWRAE 1D list may need to be converted to array with unit32
                                # MEASURE TRAINING TIME
                                train_start = time.time()
                                clf_model.fit(X_train, Y_train)
                                train_end = time.time()
                                train_time = train_end - train_start

                                inference_start = time.time()
                                Y_train_pred_prob = clf_model.predict_proba(X_train)[:, 1]
                                inference_end = time.time()
                                inference_time = inference_end - inference_start

                                total_time = hp_search_time + train_time + inference_time
                                
                                meta_info = [DATASET_SUBSET[dataset_indx], group_name, split, fold, descriptor, hp_search_time, train_time, inference_time, total_time, final_params, N_TM_EPOCHS]
                                
                                write_clf_scores(Y=Y_train, Y_pred=Y_train_pred_prob, meta_info=meta_info, model=model_label, dataset="Train", writer=result_writer)
                                Y_test_pred_prob = clf_model.predict_proba(X_test)[:, 1]
                                write_clf_scores(Y=Y_test, Y_pred=Y_test_pred_prob, meta_info=meta_info, model=model_label, dataset="Test", writer=result_writer)
                                
                                meta_info = [DATASET_SUBSET[dataset_indx], group_name, split, fold, descriptor, N_TM_EPOCHS]
                                write_MICRO_clf_scores(Y_index=train_indx, Y=Y_train, Y_pred=Y_train_pred_prob, meta_info=meta_info, model=model_label, dataset="Train", micro_writer=micro_result_writer)
                                write_MICRO_clf_scores(Y_index=test_indx, Y=Y_test, Y_pred=Y_test_pred_prob, meta_info=meta_info, model=model_label, dataset="Test", micro_writer=micro_result_writer)     
                            else:
                                final_params["number_of_clauses"] = N_CLAUSES
                                final_params["weighted_clauses"] = True
                                
                                clf_model = MultiClassTsetlinMachine(**final_params)

                                tm_train_time = 0
                                tm_inference_time = 0
                                for epoch in range(N_TM_EPOCHS):
                                    
                                    tm_train_start = time.time()
                                    clf_model.fit(X_train, Y_train, incremental=True, epochs=1)
                                    tm_train_end = time.time()
                                    
                                    tm_train_time = tm_train_time + (tm_train_end - tm_train_start)
                                
                                    tm_inference_start = time.time()
                                    Y_train_ccs = parallel_tm_ccs(tm=clf_model, X=X_train, n_clauses=N_CLAUSES)
                                    tm_inference_end = time.time()
                                    tm_inference_time = tm_inference_time + (tm_inference_end - tm_inference_start)
                                    
                                    tm_total_time = hp_search_time + tm_train_time + tm_inference_time
                                    
                                    meta_info = [DATASET_SUBSET[dataset_indx], group_name, split, fold, descriptor, hp_search_time, tm_train_time, tm_inference_time, tm_total_time, final_params, epoch+1]
                                    
                                    Y_train_pred_prob = expit(Y_train_ccs/clf_model.T)[1]
                                    
                                    write_clf_scores(Y=Y_train, Y_pred=Y_train_pred_prob, meta_info=meta_info, model=model_label, dataset="Train", writer=result_writer)
                    
                                    Y_test_ccs = parallel_tm_ccs(tm=clf_model, X=X_test, n_clauses=N_CLAUSES)
                                    Y_test_pred_prob = expit(Y_test_ccs/clf_model.T)[1]
                                    write_clf_scores(Y=Y_test, Y_pred=Y_test_pred_prob, meta_info=meta_info, model=model_label, dataset="Test", writer=result_writer)
                                    
                                    meta_info = [DATASET_SUBSET[dataset_indx], group_name, split, fold, descriptor, epoch+1] 
                                    write_MICRO_clf_scores(Y_index=train_indx, Y=Y_train, Y_pred=Y_train_pred_prob, meta_info=meta_info, model=model_label, dataset="Train", micro_writer=micro_result_writer)
                                    write_MICRO_clf_scores(Y_index=test_indx, Y=Y_test, Y_pred=Y_test_pred_prob, meta_info=meta_info, model=model_label, dataset="Test", micro_writer=micro_result_writer)                          
result_file.close()
micro_result_file.close()