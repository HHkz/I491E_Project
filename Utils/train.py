from tqdm.notebook import tqdm as tqdm 
import copy
import torch
import time

def train_step(model, features, labels):

    labels = labels.to(model.device)
    model.train()

    model.optimizer.zero_grad()

    predictions = model(features)
    loss = model.loss_func(predictions, labels)
    metric = model.metric_func(predictions, labels)

    loss.backward()
    model.optimizer.step()
    
    del predictions

    return loss.item(), metric.item()

@torch.no_grad()
def valid_step(model, features, labels):

    labels = labels.to(model.device)
    model.eval()

    predictions = model(features)
    loss = model.loss_func(predictions, labels)
    metric = model.metric_func(predictions, labels)
    
    del predictions

    return loss.item(), metric.item()

def train_classify_model(
    model, 
    epochs, 
    dl_train, 
    dl_valid, 
    save_best = None, 
):
    if epochs == 0:
        return

    metric_name = model.metric_name
    bar_format = '{percentage:3.0f}%|{n_fmt}/{total_fmt}[{elapsed}<{remaining}{postfix}]'
    best_model = None
    best_res = 0
    total_step = 0
    
    if save_best is not None:
        if save_best == 'loss':
            best_res = 16385

    for epoch in range(1, epochs+1):  
        
        start_time = time.time()
        
        torch.cuda.empty_cache()
        loss_sum = 0.0
        correct_sum = 0.0
        total_num = 0
        step = 1

        print('Epoch {:>3d} / {:>3d}'.format(epoch, epochs))

        tqdm_train = tqdm(total=len(dl_train), bar_format=bar_format)
        for step, (features, labels) in enumerate(dl_train, 1):

            loss, metric = train_step(model, features, labels)

            loss_sum += loss * labels.shape[0]
            correct_sum += metric * labels.shape[0]
            total_num += labels.shape[0]
            loss_mean = loss_sum / total_num
            metric_mean = correct_sum / total_num
            
            total_step += 1

            tqdm_train.update(1)
            tqdm_train.set_postfix({'loss':'{0:1.4f}'.format(loss_mean), 'accuracy': '{0:1.4f}'.format(metric_mean)})
            
        train_time = time.time() - start_time
        
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_correct_sum = 0
        val_total_sum = 0
        val_step = 1

        for val_step, (features,labels) in enumerate(dl_valid, 1):

            val_loss, val_metric = valid_step(model, features, labels)

            val_loss_sum += val_loss * labels.shape[0]
            val_correct_sum += val_metric * labels.shape[0]
            val_total_sum += labels.shape[0]

        val_loss_mean = val_loss_sum / val_total_sum
        val_metric_sum = val_correct_sum / val_total_sum

        print('val_loss:{0:1.4f}, val_acc:{1:1.4f}'.format(val_loss_mean, val_metric_sum))
        
        epoch_time = time.time() - start_time
        
        if save_best is not None:
            if save_best == 'acc':
                if val_metric_sum > best_res:
                    best_res = val_metric_sum
                    best_model = copy.deepcopy(model)
                    print('>>>>>>>>>>>>>>>>>>>>>>>Best result:{0:1.4f}>>>>>>>>>>>>>>>>>>>>>>>'.format(val_metric_sum))
            elif save_best == 'loss':
                if val_loss_mean < best_res:
                    best_res = val_loss_mean
                    best_model = copy.deepcopy(model)
                    print('>>>>>>>>>>>>>>>>>>>>>>>Best result:{0:1.4f}>>>>>>>>>>>>>>>>>>>>>>>'.format(val_loss_mean))

    if save_best is not None:
        for name, param in best_model.state_dict().items():
            model.state_dict()[name].copy_(param)
        del best_model