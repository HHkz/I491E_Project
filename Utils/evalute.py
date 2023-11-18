import torch
import numpy as np
from Utils.dataset_processing import evalute_prompting
from tqdm.notebook import tqdm as tqdm 
import math
from torcheval.metrics.functional import multiclass_f1_score

def ICLAcc_evalute(
        model, 
        tokenizer, 
        dataset, 
        demos_amount, 
        tries=1, 
    ):
    torch.cuda.empty_cache()
    total_count = 0
    correct_count = 0
    bar_format = '{percentage:3.0f}%|{n_fmt}/{total_fmt}[{elapsed}<{remaining}{postfix}]'
    tqdm_ICL = tqdm(total=len(dataset), bar_format=bar_format)
    
    true_labels = []
    predicted_labels = []
    
    for i in range(0, len(dataset)):
        for j in range(0, tries):
            prompt, true_label = evalute_prompting(dataset, demos_amount, i)
            tokenized_input = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            result_vector = model(tokenized_input)['logits'][0][-1].cpu().detach().numpy()
            total_count += 1
            label_space_p = []
            for labels in [6374, 8178, 21104]:
                label_space_p.append(result_vector[labels])
            label_map = {"positive": 0, "negative": 1, "neutral": 2}
            true_label_index = label_map[true_label]
            if true_label_index == np.argmax(label_space_p):
                correct_count += 1
            
            true_labels.append(true_label_index)
            predicted_labels.append(np.argmax(label_space_p))
            
            MF1 = multiclass_f1_score(torch.tensor(predicted_labels), torch.tensor(true_labels), num_classes = 3, average = 'macro')
            
            del tokenized_input
            del label_space_p
            del result_vector
            tqdm_ICL.set_postfix({
                'accuracy': '{0:1.4f}'.format(correct_count / total_count),
                'MF1': '{0:1.4f}'.format(MF1)})
        tqdm_ICL.update(1)
    
    return correct_count / total_count
