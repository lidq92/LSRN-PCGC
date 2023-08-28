import torch
from ignite.utils import convert_tensor
from ignite.engine.engine import Engine


def _prepare_batch(batch, device=None, non_blocking=False):
    x, (y, points, name) = batch 
    
    return (convert_tensor(x[0], device=device, non_blocking=non_blocking), 
            (convert_tensor(y[0], device=device, non_blocking=non_blocking), 
            points[0], name[0]))


def create_supervised_trainer(model, optimizer, loss_fn, device=None, div_steps=200, 
                              non_blocking=False, prepare_batch=_prepare_batch):
    if device: model.to(device)

    def _update(engine, batch):
        model.train()
        x, (y, points, name) = prepare_batch(batch, device=device, non_blocking=non_blocking)
        
        bs = x.shape[0] // div_steps
        avg_loss = 0.
        for d in range(div_steps):
            y_pred = model(x[d*bs:(d+1)*bs])
            loss = loss_fn(y_pred, y[d*bs:(d+1)*bs])
            avg_loss += loss.item()*y_pred.shape[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        return avg_loss / x.shape[0] #

    return Engine(_update)


def create_supervised_evaluator(model, metrics, device = None,
                                non_blocking = False, prepare_batch = _prepare_batch,
                                output_transform = lambda x, y, y_pred: (y_pred, y)):
    metrics = metrics or {}
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, (y, points, name) = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            
            return output_transform(x, (y, points, name), y_pred)

    evaluator = Engine(_inference)
    for metric_name, metric in metrics.items():
        metric.attach(evaluator, metric_name)

    return evaluator
