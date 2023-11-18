import copy
import torch

def model_merge(model_list):
    for model in model_list:
        for param in model.parameters():
            param.requires_grad = False

    return_model = copy.deepcopy(model_list[0])
    ret_dict = return_model.state_dict()

    for name, value in return_model.named_parameters():
        ret_dict[name] = torch.zeros_like(ret_dict[name])
        for model in model_list:
            ret_dict[name] += model.state_dict()[name] / len(model_list)
    return_model.load_state_dict(ret_dict)

    return return_model