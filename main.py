import openai
import json
import numpy as np
import pandas as pd
import argparse
from utils import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from model import *
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="torch.set_default_tensor_type() is deprecated as of PyTorch 2.1")

# Set your OpenAI API key
openai.api_key = ''


def knowledge_tracing(args, question_ids, kc_ids, answer_sequence, option_sequence, option_weights, next_question_id, next_kc_id):

    question_ids = [int(q_id) for q_id in question_ids]
    kc_ids = [kc_id for kc_id in kc_ids]

    option_sequence = [opt for opt in option_sequence]
    option_weights = [wt for wt in option_weights]
    next_question_id = int(next_question_id)
    next_kc_id = next_kc_id
    
    format_dict = {
        "question_ids": question_ids,
        "kc_ids": kc_ids,
        "answer_sequence": answer_sequence,
        "next_question_id": next_question_id,
        "next_kc_id": next_kc_id
    }

    if args.prompt_mode == 'option':
        format_dict["option_sequence"] = option_sequence
        
    elif args.prompt_mode == 'weight2' or 'weight':
        format_dict["option_sequence"] = option_sequence
        format_dict["option_weights"] = option_weights

    system_prompt = load_prompt_from_file(os.path.join(args.prompt_file, f'system_prompt_{args.prompt_mode}.txt'))
    user_prompt = load_prompt_from_file(os.path.join(args.prompt_file, f'prompt_{args.prompt_mode}.txt'))
    
    user_prompt = format_prompt(user_prompt, **format_dict)
    
    model = Model(args.model_name, args.temp)
    response_text = model.ask_chatgpt([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
  
    return response_text

def save_results_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Knowledge Tracing using GPT")
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo', help='Model name to use')
    parser.add_argument('--temp', type=float, default=0.0, help='Temperature setting for the model')
    parser.add_argument('--dataset_name', type=str, default='eedi_a', help='Dataset name')
    parser.add_argument('--weight_mode', type=str, default='text', help='Mode for generating option weights (e.g., text, int)')
    parser.add_argument('--task', type=str, default='Task1', help='Task type (e.g., Task1, Task2)')
    parser.add_argument('--prompt_mode', type=str, default='weight', help='prompt type (e.g., base, option, weight)')
    parser.add_argument('--prompt_file', type=str, default='prompt', help='Path to the prompt template text file')

    args = parser.parse_args()
    
    accs_per_seed, f1s_per_seed, precisions_per_seed, recalls_per_seed = [], [], [], []

    if args.task == "Task1":
        max_length_values = range(5,30,5)
        few_shot_values = [1]
    else: 
        max_length_values = [100]
        few_shot_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]

    all_results = []
    all_max_length_results = []

    for max_len in max_length_values:
        for few_shot in few_shot_values:
            accs_per_seed, f1s_per_seed, precisions_per_seed, recalls_per_seed = [], [], [], []
            args.fewshot = few_shot
            args.max_length = max_len

            for seed in range(5):
                test_dataset, w_e_o_t = (load_dataset_Task2(args, seed, args.dataset_name)
                                         if args.task == 'Task2'
                                         else load_dataset_Task1(args, seed, args.dataset_name))

                all_response, y_true = [], []

                for step, batch in enumerate(test_dataset):
                    question_ids, kc_ids, answer_sequence, option_sequence, option_weights = [], [], [], [], []
                    
                    try:
                        for i in range(len(batch[0])):
                            if batch[0][i] == -1: 
                                break
                            question_ids.append(int(batch[0][i]))
                            kc_ids.append(batch[3][i])
                            option_sequence.append(batch[2][i])
                            answer_sequence.append('correct' if batch[1][i] == 1 else 'wrong')
                            option_weights.append(w_e_o_t.loc[batch[0][i], batch[2][i]])

                        result = knowledge_tracing(
                            args, question_ids[:-1], kc_ids[:-1], answer_sequence[:-1], 
                            option_sequence[:-1], option_weights[:-1], question_ids[-1], kc_ids[-1]
                        )

                        if result in ["correct", "wrong", "incorrect", np.nan]:
                            all_response.append(result)
                            y_true.append(answer_sequence[-1])

                    except:
                        pass

                y_pred = [1 if res == 'correct' else 0 for res in all_response]
                y_true_bin = [1 if true == 'correct' else 0 for true in y_true]

                acc = accuracy_score(y_true_bin, y_pred)
                f1 = f1_score(y_true_bin, y_pred, zero_division=0)
                precision = precision_score(y_true_bin, y_pred, zero_division=0)
                recall = recall_score(y_true_bin, y_pred, zero_division=0)

                accs_per_seed.append(acc)
                f1s_per_seed.append(f1)


                result_row = [seed, max_len, acc, f1]
                all_max_length_results.append(result_row)

            df_per_seed = pd.DataFrame(all_max_length_results, columns=['seed', 'Max_length', 'ACC_mean', 'F1_mean'])

            acc_mean, acc_var = np.mean(accs_per_seed), np.std(accs_per_seed)
            f1_mean, f1_var = np.mean(f1s_per_seed), np.std(f1s_per_seed)
 

            df_overall = pd.DataFrame([[
                'Overall', max_len,
                acc_mean, acc_var,
                f1_mean, f1_var
            ]], columns=['seed', 'Max_length', 'ACC_mean', 'ACC_var', 'F1_mean', 'F1_var'])

            df_overall.to_csv(f'{args.dataset_name}_{args.weight_mode}_{args.prompt_mode}_{max_len}_{args.task}_{args.model_name}_overall.csv', index=False)

            print(f"Max_length={max_len} Few_shot={few_shot}| Accuracy: mean={acc_mean:.4f}, var={acc_var:.4f} | "
                  f"F1: mean={f1_mean:.4f}, var={f1_var:.4f} ")
