import os
from tqdm import tqdm
import copy
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import sklearn.metrics
from transformers import AutoTokenizer
import torch
import sklearn
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
from models.sinkhorn_knopp import SinkhornKnopp

import numpy as np

def cosine_similarity_matrix(A, B):
    # A: [N, D], B: [M, D]
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)  # [N, D]
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)  # [M, D]

    similarity = A_norm @ B_norm.T  # [N, M]
    return similarity

def hungarian_alignment_with_unlabeled(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 获取有标签样本索引
    labeled_mask = y_true != -1
    y_true_labeled = y_true[labeled_mask]
    y_pred_labeled = y_pred[labeled_mask]

    # 步骤 1：构造权重矩阵，用于匈牙利算法
    D = max(y_pred.max(), y_true_labeled.max()) + 1
    w = np.zeros((D, D), dtype=np.int32)
    for i in range(y_pred_labeled.size):
        w[y_pred_labeled[i], y_true_labeled[i]] += 1

    # 步骤 2：匈牙利算法，返回最佳匹配对
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    cluster_to_label = {cluster: label for cluster, label in zip(row_ind, col_ind)}

    # 步骤 3：为未匹配的聚类 id 分配新的标签 id
    used_labels = set(col_ind.tolist())
    next_new_label = max(used_labels) + 1 if used_labels else 0
    all_cluster_ids = set(y_pred.tolist())
    unmatched_clusters = all_cluster_ids - set(cluster_to_label.keys())

    for cluster in unmatched_clusters:
        cluster_to_label[cluster] = next_new_label
        next_new_label += 1

    # 步骤 4：将原始 y_pred 映射为新标签
    mapped_pred = np.array([cluster_to_label[cid] for cid in y_pred])

    return mapped_pred, cluster_to_label

def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred, known_lab):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    ind_map = {j: i for i, j in ind.tolist()}
    
    old_acc = 0
    total_old_instances = 0
    for i in known_lab:
        if i not in ind_map:
            print(i)
            print(ind_map)
            np.save('../outputs/y_pred.npy', y_pred)
            np.save('../outputs/y_true.npy', y_true)
        
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances
    
    new_acc = 0
    total_new_instances = 0
    for i in range(w.shape[0]):
        if i not in known_lab:
            new_acc += w[ind_map[i], i]
            total_new_instances += sum(w[:, i])
    if total_new_instances != 0:
        new_acc /= total_new_instances

    h_score = 2*old_acc*new_acc / (old_acc + new_acc)

    metrics = {
        'ACC': round(acc*100, 2),
        'H-Score': round(h_score*100, 2),
        'K-ACC': round(old_acc*100, 2),
        'N-ACC': round(new_acc*100, 2),
    }

    return metrics, ind

def clustering_score(y_true, y_pred, known_lab):
    metrics, ind = clustering_accuracy_score(y_true, y_pred, known_lab)
    metrics['ARI'] = round(adjusted_rand_score(y_true, y_pred)*100, 2)
    metrics['NMI'] = round(normalized_mutual_info_score(y_true, y_pred)*100, 2)
    return metrics, ind

class Metrics():
    def __init__(self, data_args, model_args, training_args):
        self.known_labels = pd.read_csv(f'../data/{data_args.dataset_name}/label/label_{data_args.rate}.list', header=None)[0].tolist()
        self.ori_labels = pd.read_csv(f'../data/{data_args.dataset_name}/label/label.list', header=None)[0].tolist()
        self.ood_labels = [i for i in self.ori_labels if i not in self.known_labels]
        self.ori_labels = self.known_labels + self.ood_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        self.vector_dir = data_args.vector_dir
        self.mode = model_args.mode
        self.data_args = data_args

        self.train_data = pd.read_csv(f'../data/{data_args.dataset_name}/labeled_data/train_{data_args.labeled_ratio}.tsv', sep='\t')
        self.dev_data = pd.read_csv(f'../data/{data_args.dataset_name}/labeled_data/dev_{data_args.labeled_ratio}.tsv', sep='\t')
        self.test_data = pd.read_csv(f'../data/{data_args.dataset_name}/test.tsv', sep='\t')

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.class_tokens = self.tokenizer(
                self.ori_labels,
                truncation=True,
                padding=True,
                add_special_tokens=True,
                max_length=512,
                return_overflowing_tokens=False,
                return_length=False,
            )['input_ids']
        self.sinkhorn = SinkhornKnopp(self.data_args)
    
    def compute_metrics(self, eval_preds, compute_result=True):
        (token_preds, mlp_class_logits, com_class_logits, class_hidden_states, mlp_hidden_states, com_hidden_states, token_type_ids, class_golds), labels = eval_preds

        shift_token_preds = token_preds[:,:-1]
        shift_token_type_ids = token_type_ids[:,1:]

        preds = [self.tokenizer.decode(shift_token_preds[i][shift_token_type_ids[i] > 1]).strip(' \n</s>') for i in range(shift_token_preds.shape[0])]
        golds = [self.ori_labels[i] if i != -1 else 'ood' for i in class_golds]

        df = pd.DataFrame({'preds': preds, 'golds': golds})
        df['gold_ids'] = class_golds.tolist()
        df['preds'] = df['preds'].apply(lambda x: x.replace('\n', '.'))

        known_lab = df[df['golds'].isin(self.known_labels)]['gold_ids'].unique()

        all_hidden_states = {
            "class": class_hidden_states,
            "mlp": mlp_hidden_states,
            "com": com_hidden_states,
        }

        ## test the metrics
        all_metrics = {}
        prev_labels = copy.deepcopy(df['gold_ids'].values)
        prev_labels[prev_labels >= len(self.known_labels)] = -1
        for key, values in all_hidden_states.items():
            kmeans = KMeans(n_clusters=len(self.ori_labels), n_init=5)
            kmeans.fit(values)
            ori_preds = kmeans.predict(values)

            cluster_centers_ = kmeans.cluster_centers_
            logits = cosine_similarity_matrix(values, cluster_centers_)
            sk_logits = self.sinkhorn(torch.tensor(logits)).numpy()
            sk_preds = np.argmax(sk_logits, axis=-1)

            all_preds = {
                "kmeans": ori_preds,
                "sk-kmeans": sk_preds,
            }

            for j, preds in all_preds.items():
                ans, w = hungarian_alignment_with_unlabeled(prev_labels, preds)
                prev_labels = ans
                df[f'pred_cluster_{key}_ids'] = ans
                metric_result, ind = clustering_score(df['gold_ids'].values, preds, known_lab)
                for i in metric_result:
                    all_metrics[f'{j}_{key}_{i}'] = metric_result[i]

            if self.mode == 'eval-test' and key == 'mlp':
                torch.save(values, f"{self.data_args.vector_dir}/{self.mode}_mlp_hidden_states.pt")
                torch.save(cluster_centers_, f"{self.data_args.vector_dir}/{self.mode}_mlp_cluster_centers_.pt")
                torch.save(all_preds, f"{self.data_args.vector_dir}/{self.mode}_mlp_preds.pt")

        ## test the logits metrics
        all_class_logits = {
            "mlp": mlp_class_logits,
            "com": com_class_logits
        }

        for key, values in all_class_logits.items():

            ori_preds = values.argmax(axis=-1)
            sk_preds = self.sinkhorn(torch.tensor(values)).numpy().argmax(axis=-1)
            all_preds = {
                "logits": ori_preds,
                "sk-logits": sk_preds,
            }
            for j, preds in all_preds.items():
                metric_result, ind = clustering_score(df['gold_ids'].values, preds, known_lab)
                for i in metric_result:
                    all_metrics[f'{j}_{key}_{i}'] = metric_result[i]

        return all_metrics
    
    def preprocess_logits_for_metrics(self, logits, labels):
        mlp_class_logits, com_class_logits = None, None
        if len(logits) == 9:
            ori_logits, mlp_class_logits, com_class_logits, past_values, class_hidden_states, mlp_hidden_states, com_hidden_states, token_type_ids, class_golds = logits
        elif len(logits) == 8:
            ori_logits, mlp_class_logits, com_class_logits, class_hidden_states, mlp_hidden_states, com_hidden_states,  token_type_ids, class_golds = logits
        else:
            assert False

        return (ori_logits.max(dim=-1).indices, mlp_class_logits, com_class_logits, class_hidden_states, mlp_hidden_states, com_hidden_states, token_type_ids, class_golds)
