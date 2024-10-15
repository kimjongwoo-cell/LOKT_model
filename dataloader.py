import os
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="torch.set_default_tensor_type() is deprecated as of PyTorch 2.1")
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


class load_dataset(Dataset):
    def __init__(self, seq_len, dataset_dir):
        super().__init__()
        self.dataset_dir = 'data/' + dataset_dir
        self.dataset_path = f"data/{dataset_dir}/{dataset_dir}.pkl"
        
        if os.path.exists(os.path.join(self.dataset_dir, "q_seqs.pkl")):
            with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "o_seqs.pkl"), "rb") as f:
                self.o_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "kc_seqs.pkl"), "rb") as f:
                self.kc_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q_list.pkl"), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u_list.pkl"), "rb") as f:
                self.u_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "user_ids.pkl"), "rb") as f:
                self.user_ids = pickle.load(f)
        else:
            self.q_seqs, self.r_seqs, self.o_seqs, self.kc_seqs, self.q_list, self.u_list, self.user_ids = self.preprocess()

        self.num_q = max(max(sub_lst) for sub_lst in self.q_seqs) + 1

        if seq_len:
            self.q_seqs, self.r_seqs, self.o_seqs, self.kc_seqs, self.user_ids = \
                match_seq_len(self.q_seqs, self.r_seqs, self.o_seqs, self.kc_seqs, self.user_ids, seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index], self.o_seqs[index], self.kc_seqs[index], self.user_ids[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        df = pd.read_pickle(self.dataset_path)
        if 'ednet' in self.dataset_path:
            user_id_col, question_id_col = 'user_id', 'question_id'
            is_correct_col, answer_value_col = 'corretness', 'user_answer'
            kcs = 'tags'
        elif 'eedi' in self.dataset_path:
            user_id_col, question_id_col = 'UserId', 'QuestionId'
            is_correct_col, answer_value_col = 'IsCorrect', 'AnswerValue'
            kcs = 'tag'
        elif 'assist' in self.dataset_path:
            user_id_col, question_id_col = 'user_id', 'problem_id'
            is_correct_col, answer_value_col = 'correct', 'answer_id'
            kcs = 'skill_id'
        elif 'DBE' in self.dataset_path:
            user_id_col, question_id_col = 'student_id', 'question_id'
            is_correct_col, answer_value_col = 'answer_state', 'answer_choice_id'
            kcs = 'knowledgecomponent_id'
            
        u_list = np.unique(df[user_id_col].values)
        q_list = np.unique(df[question_id_col].values)

        q_seqs, r_seqs, o_seqs, kc_seqs, user_ids = [], [], [], [], []

        grouped = df.groupby(df.iloc[:, 0])
        for student, group in grouped:
            q_seqs.append(group.iloc[:, 1].values)
            r_seqs.append(group.loc[:, is_correct_col].values)
            o_seqs.append(group.loc[:, answer_value_col].values)
            kc_seqs.append(group.loc[:, kcs].values.tolist())
            user_ids.append(student)
        
        with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "wb") as f:
            pickle.dump(q_seqs, f)
        with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.dataset_dir, "o_seqs.pkl"), "wb") as f:
            pickle.dump(o_seqs, f)
        with open(os.path.join(self.dataset_dir, "q_list.pkl"), "wb") as f:
            pickle.dump(q_list, f)
        with open(os.path.join(self.dataset_dir, "u_list.pkl"), "wb") as f:
            pickle.dump(u_list, f)
        with open(os.path.join(self.dataset_dir, "kc_seqs.pkl"), "wb") as f:
            pickle.dump(kc_seqs, f)
        with open(os.path.join(self.dataset_dir, "user_ids.pkl"), "wb") as f:
            pickle.dump(user_ids, f)
        
        return q_seqs, r_seqs, o_seqs, kc_seqs, q_list, u_list, user_ids


def match_seq_len(q_seqs, r_seqs, o_seqs, kc_seqs, user_ids, seq_len, pad_val=-1):
    
    proc_q_seqs, proc_r_seqs, proc_o_seqs, proc_kc_seqs, proc_user_ids = [], [], [], [], []

    for q_seq, r_seq, o_seq, kc_seq, user_id in zip(q_seqs, r_seqs, o_seqs, kc_seqs, user_ids):
        i = 0
        
        while i + seq_len + 1 < len(q_seq):
            proc_q_seqs.append(q_seq[i:i + seq_len + 1])
            proc_r_seqs.append(r_seq[i:i + seq_len + 1])
            proc_o_seqs.append(o_seq[i:i + seq_len + 1])
            proc_kc_seqs.append(kc_seq[i:i + seq_len + 1])
            proc_user_ids.append(user_id)
            i += seq_len + 1

        proc_q_seqs.append(np.concatenate([q_seq[i:], np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))]))
        proc_r_seqs.append(np.concatenate([r_seq[i:], np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))]))
        proc_o_seqs.append(np.concatenate([o_seq[i:], np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))]))
        
        if isinstance(kc_seq[0], list):
            proc_kc_seqs.append(kc_seq[i:] + [[pad_val]] * (i + seq_len + 1 - len(q_seq)))
        else:    
            proc_kc_seqs.append(np.concatenate([kc_seq[i:], np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))]))
        
        proc_user_ids.append(user_id)

    return proc_q_seqs, proc_r_seqs, proc_o_seqs, proc_kc_seqs, proc_user_ids
