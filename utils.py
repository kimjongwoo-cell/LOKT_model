import numpy as np
import torch
import pandas as pd
import pickle
import random
from dataloader import * 
import warnings
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split, Subset
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="torch.set_default_tensor_type() is deprecated as of PyTorch 2.1")

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)


def load_prompt_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def format_prompt(prompt, **kwargs):
    try:
        return prompt.format(**kwargs)
    except KeyError as e:
        print(f"Missing key in format: {e}")
        return prompt

def match_seq_len(q_seqs, r_seqs, o_seqs, kc_seqs, seq_len, pad_val=-1):
    proc_q_seqs, proc_r_seqs, proc_o_seqs, proc_kc_seqs = [], [], [], []
    for q_seq, r_seq, o_seq, kc_seq in zip(q_seqs, r_seqs, o_seqs, kc_seqs):
        i = 0
        while i + seq_len + 1 < len(q_seq):
            proc_q_seqs.append(q_seq[i:i + seq_len + 1])
            proc_r_seqs.append(r_seq[i:i + seq_len + 1])
            proc_o_seqs.append(o_seq[i:i + seq_len + 1])
            proc_kc_seqs.append(kc_seq[i:i + seq_len + 1])
            i += seq_len + 1
        pad_length = i + seq_len + 1 - len(q_seq)
        proc_q_seqs.append(np.concatenate([q_seq[i:], np.full(pad_length, pad_val)]))
        proc_r_seqs.append(np.concatenate([r_seq[i:], np.full(pad_length, pad_val)]))
        proc_o_seqs.append(np.concatenate([o_seq[i:], np.full(pad_length, pad_val)]))
        proc_kc_seqs.append(np.concatenate([kc_seq[i:], np.full(pad_length, pad_val)]))
    return proc_q_seqs, proc_r_seqs, proc_o_seqs, proc_kc_seqs

def collate_fn(batch, pad_val=-1):
    q_seqs, r_seqs, o_seqs, kc_seqs = [], [], [], []
    qshft_seqs, rshft_seqs, oshft_seqs, kcshft_seqs = [], [], [], []
    for q_seq, r_seq, o_seq, kc_seq in batch:
        q_seqs.append(torch.FloatTensor(q_seq[:-1]))
        r_seqs.append(torch.FloatTensor(r_seq[:-1]))
        o_seqs.append(torch.FloatTensor(o_seq[:-1]))
        kc_seqs.append(torch.FloatTensor(kc_seq[:-1]))
        qshft_seqs.append(torch.FloatTensor([q_seq[-1]]))
        rshft_seqs.append(torch.FloatTensor([r_seq[-1]]))
        oshft_seqs.append(torch.FloatTensor([o_seq[-1]]))
        kcshft_seqs.append(torch.FloatTensor([kc_seq[-1]]))
    q_seqs = pad_sequence(q_seqs, batch_first=True, padding_value=pad_val)
    r_seqs = pad_sequence(r_seqs, batch_first=True, padding_value=pad_val)
    o_seqs = pad_sequence(o_seqs, batch_first=True, padding_value=pad_val)
    kc_seqs = pad_sequence(kc_seqs, batch_first=True, padding_value=pad_val)
    return q_seqs, r_seqs, o_seqs, kc_seqs, qshft_seqs, rshft_seqs, oshft_seqs, kcshft_seqs

def replace_neg_ones(arr):
    valid_indices = np.where(arr != -1)[0]
    if len(valid_indices) > 0:
        last_valid_index = valid_indices[-1]
        arr[arr == -1] = arr[last_valid_index]
    return arr

def sample_train_dataset(train_dataset, num_samples=8, seed=0):
    samples = [train_dataset[i] for i in range(len(train_dataset))]
    indices_last_zero = [i for i, sample in enumerate(samples) if replace_neg_ones(sample[1])[-1] == 0]
    indices_last_one = [i for i, sample in enumerate(samples) if replace_neg_ones(sample[1])[-1] == 1]
    random.seed(seed)
    samples_zero = random.sample(indices_last_zero, num_samples)
    samples_one = random.sample(indices_last_one, num_samples)
    random_indices = samples_zero + samples_one
    random_samples = [train_dataset[i] for i in random_indices]
    return random_samples

def generate_option_weight(data_name, df, mode='int'):
    if data_name == 'eedi_b':
        pickle_file_path = r"data\eedi_b\eedi_b.pkl"
        user_id_col, question_id_col = 'UserId', 'QuestionId'
        is_correct_col, answer_value_col = 'IsCorrect', 'AnswerValue'
    elif data_name == 'eedi_a':
        pickle_file_path = r"data\eedi_a\eedi_a.pkl"
        user_id_col, question_id_col = 'UserId', 'QuestionId'
        is_correct_col, answer_value_col = 'IsCorrect', 'AnswerValue'
    elif data_name == 'ednet':
        pickle_file_path = r"data\ednet\ednet.pkl"
        user_id_col, question_id_col = 'user_id', 'question_id'
        is_correct_col, answer_value_col = 'corretness', 'user_answer'
    elif data_name == 'assistment09':
        pickle_file_path = r"data\assistment09\assistment09.pkl"
        user_id_col, question_id_col = 'user_id', 'problem_id'
        is_correct_col, answer_value_col = 'correct', 'answer_id'
    elif data_name == 'DBE':
        pickle_file_path = r"data\DBE\DBE.pkl"
        user_id_col, question_id_col = 'student_id', 'question_id'
        is_correct_col, answer_value_col = 'answer_state', 'answer_choice_id'
    
    if data_name != 'ednet':
        with open(pickle_file_path, 'rb') as file:
            df_1 = pickle.load(file)
        q_id = np.max(df_1[question_id_col].values.tolist())
    else:
        q_id = 13168
    if isinstance(df, pd.DataFrame):
        df = df
        user_id_col = 'UserId'
        question_id_col = 'QuestionId'
        is_correct_col = 'IsCorrect'
        answer_value_col = 'AnswerValue'
    else:
        df = df_1    
    df = df.dropna().reset_index(drop=True)
    exercise_difficulty = df.groupby(question_id_col)[is_correct_col].mean().reset_index()
    exercise_difficulty.columns = [question_id_col, 'd_e']
    df = df.merge(exercise_difficulty, on=question_id_col, how='left')

    all_users = df[user_id_col].unique()
    correct_answers = df[df[is_correct_col] == 1].groupby(user_id_col)[is_correct_col].count().reindex(all_users, fill_value=0).reset_index()
    correct_answers.columns = [user_id_col, 'N_s']
    df = df.merge(correct_answers, on=user_id_col, how='left')

    sum_difficulties = df.groupby(user_id_col)['d_e'].sum().reset_index()
    sum_difficulties.columns = [user_id_col, 'sum_d_e']
    df = df.merge(sum_difficulties, on=user_id_col, how='left')

    num_exercises = df.groupby(user_id_col)[question_id_col].nunique().reset_index()
    num_exercises.columns = [user_id_col, 'E_s']
    df = df.merge(num_exercises, on=user_id_col, how='left')

    df['x'] = (df['N_s'] - df['sum_d_e']) / df['E_s']
    user_scores = df[[user_id_col, 'x']].drop_duplicates().reset_index(drop=True)

    E = df[question_id_col].unique()
    O = df[answer_value_col].unique()

    x_bar = df['x'].mean()
    S_x = df['x'].std()

    c_e_o = df.groupby([question_id_col, answer_value_col]).size().unstack(fill_value=0)
    total_counts = df.groupby(question_id_col).size()
    c_e_o = c_e_o.div(total_counts, axis=0)
    nc_e_o = 1 - c_e_o

    all_users = df.groupby(question_id_col)[user_id_col].apply(set).to_dict()
    s_e_o = df.groupby([question_id_col, answer_value_col])[user_id_col].apply(list).unstack(fill_value=[])

    result = {}
    for QuestionId in E:
        result[QuestionId] = {}
        for option in O:
            result[QuestionId][option] = list(all_users[QuestionId] - set(s_e_o.at[QuestionId, option]))
    ns_e_o = pd.DataFrame(result).T

    user_score_dict = user_scores.set_index(user_id_col)['x'].to_dict()

    x_c_e_o = {}
    for QuestionId in E:
        x_c_e_o[QuestionId] = {}
        for option in O:
            UserIds = s_e_o.at[QuestionId, option]
            c_e_o_set = len(UserIds)
            if UserIds:
                x_i = np.array([user_score_dict[uid] for uid in UserIds])
                numerator = np.sum(x_i)
                x_c_e_o[QuestionId][option] = numerator / c_e_o_set
            else:
                x_c_e_o[QuestionId][option] = np.nan

    x_c_e_o = pd.DataFrame(x_c_e_o).T.fillna(0)

    n_x_c_e_o = {}
    for QuestionId in E:
        n_x_c_e_o[QuestionId] = {}
        for option in O:
            UserIds = ns_e_o.at[QuestionId, option]
            n_c_e_o_set = len(UserIds)
            if UserIds:
                x_i = np.array([user_score_dict[uid] for uid in UserIds])
                numerator = np.sum(x_i)
                n_x_c_e_o[QuestionId][option] = numerator / n_c_e_o_set
            else:
                n_x_c_e_o[QuestionId][option] = np.nan

    n_x_c_e_o = pd.DataFrame(n_x_c_e_o).T.fillna(0)

    w_e_o = (x_c_e_o - n_x_c_e_o) / S_x * np.sqrt(c_e_o * nc_e_o)

    if mode == 'text':
        transformed_values = w_e_o.apply(transform_row, axis=1)
        w_e_o_t = w_e_o.copy()
        for col in w_e_o.columns:
            w_e_o_t[col] = transformed_values.apply(lambda x: x[w_e_o.columns.get_loc(col)])
        final_output = w_e_o_t
    else:
        final_output = w_e_o

    if data_name == 'assistment09':
        o_id = np.max(df[answer_value_col].values.tolist())
        final_output = final_output.reindex(index=np.arange(0, q_id+1), columns=np.arange(0, o_id+1), fill_value="nan")
    elif data_name == 'DBE':
        o_id = np.max(df[answer_value_col].values.tolist())
        final_output = final_output.reindex(index=np.arange(0, q_id+1), columns=np.arange(0, o_id+1), fill_value="nan")
    else:
        final_output = final_output.reindex(np.arange(0, q_id+1), fill_value="nan")
    return final_output




def transform_row(row):
    labels = ['Proficient', 'Partial', 'Limited', 'Inadequate']
    row = np.where(row == 0, np.nan, row)
    non_zero_values = [val for val in row if not np.isnan(val)]
    sorted_indices = np.argsort(non_zero_values)[::-1]  
    transformed_row = [np.nan] * len(row) 
    for i, idx in enumerate(sorted_indices):
        if i < len(labels):  
            transformed_row[idx] = labels[i]
    return transformed_row


def load_dataset_Task2(args,seed,data):
    seq_len = 20
    dataset = load_dataset(seq_len,data)
    def replace_neg_ones(arr):
        valid_indices = np.where(arr != -1)[0]
        if len(valid_indices) > 0:
            last_valid_index = valid_indices[-1]
            arr[arr == -1] = arr[last_valid_index]
        return arr[-1]
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # Random seed generator
    random.seed(seed)
    generator = torch.Generator('cuda').manual_seed(seed)

    # Calculate dataset sizes
    dataset_size = len(dataset)
    
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size  # 남은 데이터를 테스트 세트로 사용

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    def few_shot_sample_with_labels(dataset, num_samples, replace_neg_ones):
        samples = [dataset[i] for i in range(len(dataset))]
        indices_last_zero = [i for i, sample in enumerate(samples) if replace_neg_ones(sample[1]) == 0]
        indices_last_one = [i for i, sample in enumerate(samples) if replace_neg_ones(sample[1]) == 1]

        # Check if we have enough samples
        if len(indices_last_zero) < num_samples or len(indices_last_one) < num_samples:
            raise ValueError("Not enough samples to perform few-shot sampling.")

        if num_samples <= 1:
            samples_zero = random.sample(indices_last_zero, int(num_samples*len(indices_last_zero)))
            samples_one = random.sample(indices_last_one, int(num_samples*len(indices_last_one)))
        else:
            samples_zero = random.sample(indices_last_zero, min(num_samples, len(indices_last_zero)))
            samples_one = random.sample(indices_last_one, min(num_samples, len(indices_last_one)))
        random_indices = samples_zero + samples_one
        random.shuffle(random_indices)
        random_samples = Subset(dataset, random_indices)
        
        return random_samples


    fewshot_test_count = 50

    test_dataset = few_shot_sample_with_labels(test_dataset, fewshot_test_count, replace_neg_ones)
    train_dataset = few_shot_sample_with_labels(train_dataset, args.fewshot, replace_neg_ones)

    data_list = []

    for step, batch in enumerate(train_dataset):
 
        for i in range(len(batch[0])):
            if batch[0][i] == -1: 
                break
            data_list.append({
                'UserId': batch[-1],  # 현재 user_id 할당
                'QuestionId': int(batch[0][i]),
                'AnswerValue': batch[2][i],
                'IsCorrect': 1 if batch[1][i] == 1 else 0
               
            })
        

    df = pd.DataFrame(data_list)
    return test_dataset ,generate_option_weight(args.dataset_name,df,args.weight_mode)


def load_dataset_Task1(args,seed,data):
    seq_len = 100
    dataset = load_dataset(seq_len,data)
    num_samples = args.max_length
    
    def replace_neg_ones(arr):
        valid_indices = np.where(arr != -1)[0]
        if len(valid_indices) > 0:
            last_valid_index = valid_indices[-1]
            arr[arr == -1] = arr[last_valid_index]
        return arr[-1]
    
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # Random seed generator
    random.seed(seed)
    generator = torch.Generator('cuda').manual_seed(seed)

    # Calculate dataset sizes
    dataset_size = len(dataset)
    
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size  # 남은 데이터를 테스트 세트로 사용

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    def sample_with_last_fixed(data, num_samples):
        samples = []
        for sequence, *labels in data:
            valid_indices = np.where(sequence != -1)[0]
            if len(valid_indices) == 0:
                continue
            
            last_valid_index = valid_indices[-1]
            
            #sampled_indices = valid_indices[max(0, len(valid_indices) - num_samples):-1]  # 마지막 유효 인덱스를 제외하고 선택

            sampled_indices = random.sample(list(valid_indices[:-1]), min(num_samples-1, len(valid_indices) - 1))
            
            # sequence와 labels를 동일하게 샘플링하고 마지막 요소를 추가
            selected_sequence = np.array(sequence[sampled_indices].tolist() + [sequence[last_valid_index]])
            selected_labels = []
            for label in labels:
                if not isinstance(label, list):
                    selected_labels.append(label)
                else:
                    selected_labels.append(np.array( [label[sampled_indice] for sampled_indice in sampled_indices] + [label[last_valid_index]], dtype=object))

            
            samples.append((selected_sequence, *selected_labels))
    
        return samples
    def few_shot_sample_with_labels(dataset, num_samples, replace_neg_ones):
        samples = [dataset[i] for i in range(len(dataset))]
        indices_last_zero = [i for i, sample in enumerate(samples) if replace_neg_ones(sample[1]) == 0]
        indices_last_one = [i for i, sample in enumerate(samples) if replace_neg_ones(sample[1]) == 1]


        # Check if we have enough samples
        if len(indices_last_zero) < num_samples or len(indices_last_one) < num_samples:
            raise ValueError("Not enough samples to perform few-shot sampling.")

        samples_zero = random.sample(indices_last_zero, min(num_samples, len(indices_last_zero)))
        samples_one = random.sample(indices_last_one, min(num_samples, len(indices_last_one)))
        random_indices = samples_zero + samples_one
        random.shuffle(random_indices)
        random_samples = Subset(dataset, random_indices)
        
        return random_samples
    fewshot_test_count = 50
    
    test_dataset = few_shot_sample_with_labels(test_dataset, fewshot_test_count, replace_neg_ones)
    if num_samples != 100:
        test_dataset = sample_with_last_fixed(test_dataset, num_samples)

    data_list = []

    for step, batch in enumerate(train_dataset):
 
        for i in range(len(batch[0])):
            if batch[0][i] == -1: 
                break
            data_list.append({
                'UserId': batch[-1],  # 현재 user_id 할당
                'QuestionId': int(batch[0][i]),
                'AnswerValue': batch[2][i],
                'IsCorrect': 1 if batch[1][i] == 1 else 0
               
            })
    df = pd.DataFrame(data_list)
    
    w_e_o = generate_option_weight(args.dataset_name,df,args.weight_mode)
  
    return test_dataset,w_e_o

