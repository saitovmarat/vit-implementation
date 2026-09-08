from tqdm.auto import tqdm
import torch


def train_one_epoch(
    model, train_dataloader, optimizer, 
    loss_func, metric_func, device
):
    model.train()
    total_metric = 0
    
    for X_batch, y_batch in tqdm(train_dataloader):    
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        
        loss = loss_func(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            metric = metric_func(preds, y_batch)
            total_metric = total_metric + metric
        
    avg_metric = total_metric / len(train_dataloader)
    return avg_metric
    
    
def eval_one_epoch(
    model, val_dataloader,
    metric_func, device
):
    model.eval()
    total_metric = 0
    
    for X_batch, y_batch in tqdm(val_dataloader):    
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        with torch.no_grad():
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            metric = metric_func(preds, y_batch)
            total_metric = total_metric + metric
        
    avg_metric = total_metric / len(val_dataloader)
    return avg_metric
    
    
def train(
    model, num_epochs, train_dataloader, val_dataloader,
    optimizer, loss_func, metric_func,
    device, best_model_path=None
):
    metric = {"train": [], "val": []}
    best_val_metric = 0.0
    
    model.to(device)
    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch+1}")
        
        metric_train = train_one_epoch(
            model, train_dataloader, optimizer,
            loss_func, metric_func, device
        )
        metric["train"].append(metric_train)
        print(f"Score_train: {metric_train}")
        
        metric_val = eval_one_epoch(
            model, val_dataloader, 
            metric_func, device
        )
        metric["val"].append(metric_val)
        print(f"\nScore_val: {metric_val}")
        
        if best_model_path and metric_val > best_val_metric:
            best_val_metric = metric_val
            torch.save(model.state_dict(), best_model_path)
            print(f"\nNew best model saved (metric_val={metric_val:.4f})")
    return metric