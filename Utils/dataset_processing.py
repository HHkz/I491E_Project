import json
import random
import torch
from torch.utils.data import Dataset, DataLoader


def write_jsonl(path, data):
# Reuse from the ABSA_baseline
  with open(path, "w") as jsonl_file:
    for item in data:
        json_line = json.dumps(item)  # Convert dictionary to JSON string
        jsonl_file.write(json_line + '\n')  # Write JSON line to the file


def read_jsonl_file(file_path):
# Reuse from the ABSA_baseline
    """
    Reads a JSONL (JSON Lines) file and returns a list of parsed JSON objects.

    :param file_path: The path to the JSONL file.
    :return: A list of JSON objects.
    """
    with open(file_path, encoding="utf-8") as file:
        data_str = file.read()
        data_list = [json.loads(d) for d in data_str.split('\n') if d]
    return data_list


def read_json_to_table(file_path, supervised = True):
    return_table = []
    raw_data = read_jsonl_file(file_path)
    for row in raw_data:
        for aspect in row['aspect_categories']:
            if supervised:
                return_table.append([row['sentence'], aspect[1], aspect[2], aspect[0]])
            else:
                return_table.append([row['sentence'], aspect[1], 'none', aspect[0]])
    return return_table


class dataset_loader(Dataset):
    def __init__(
        self, 
        dataset_table, 
        tokenizer
    ):
        self.SEP = tokenizer.sep_token
        self.dataset = dataset_table
        self.tokenizer = tokenizer
        self.label_map = {"positive": 0, "negative": 1, "neutral": 2}
             
    def __getitem__(self, index):
        line = self.dataset[index]
        sent = line[0]
        target = line[1]
        polarity = -1
        if line[2] != 'none':
            polarity = self.label_map[line[2]]
#        label = [0.0, 0.0, 0.0]
#        label[polarity] = 1.0
        sentence = sent + "</s>" + target + "</s>" + "positive, negative, neutral"
        sent_token = self.tokenizer(sentence, padding = 'max_length', max_length = 425)
        return ([torch.tensor(sent_token['input_ids']), torch.tensor(sent_token['attention_mask'])], torch.tensor(polarity))
    
    def __len__(self):
        return len(self.dataset)
    

def load_data(dataset, batch_size):    
    dl = DataLoader(
        dataset,
        batch_size,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
    )
    
    return dl