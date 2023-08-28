import torch
from ignite.utils import convert_tensor
from ignite.engine.engine import Engine


def _prepare_test_batch(batch, device=None, non_blocking=False):
    x, (y, points, name) = batch 
    
    return (convert_tensor(x[0], device=device, non_blocking=non_blocking), 
            (convert_tensor(y[0], device=device, non_blocking=non_blocking), 
            points[0], name[0]))


def _prepare_train_batch(batch, device=None, non_blocking=False):
    x, y = batch 
    
    return (convert_tensor(x, device=device, non_blocking=non_blocking), 
            convert_tensor(y, device=device, non_blocking=non_blocking))


def create_supervised_trainer(model, optimizer, loss_fn, device=None, 
                              non_blocking=False, prepare_batch=_prepare_train_batch):
    if device: model.to(device)

    def _update(engine, batch):
        optimizer.zero_grad()
        model.train()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
            
        return loss 

    return Engine(_update)


def create_supervised_validator(model, metrics, device = None,
                                non_blocking = False, prepare_batch = _prepare_train_batch,
                                output_transform = lambda x, y, y_pred: (y_pred, y)):
    metrics = metrics or {}
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            
            return output_transform(x, y, y_pred)

    evaluator = Engine(_inference)
    for metric_name, metric in metrics.items():
        metric.attach(evaluator, metric_name)

    return evaluator


def create_supervised_evaluator(model, metrics, device = None,
                                non_blocking = False, prepare_batch = _prepare_test_batch,
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
