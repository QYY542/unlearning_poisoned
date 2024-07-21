import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch.nn as nn

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.targets = [label for _, label in data]  # 这里假设 data 中每个元素都是 (image, label) 形式

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # 当data[idx]是(image, label)形式时直接返回


def hvp(model, x, y, v):
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x), y)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    hvp = torch.autograd.grad(grads, model.parameters(), grad_outputs=v)
    return hvp

def get_inv_hvp_lissa(model, x, y, v, hvp_batch_size, scale, damping, iterations=-1, verbose=False,
                      repetitions=1, early_stopping=True, patience=20, hvp_logger=None):
    """
    Calculate H^-1*v using the iterative scheme proposed by Agarwal et al with batch updates.
    The scale and damping parameters have to be found by trial and error to achieve convergence.
    Rounds can be set to average the results over multiple runs to decrease variance and stabilize the results.
    """
    hvp_batch_size = int(hvp_batch_size)
    n_batches = 100 * np.ceil(x.shape[0] / hvp_batch_size) if iterations == -1 else iterations
    shuffle_indices = [torch.randperm(x.shape[0]) for _ in range(repetitions)]
    
    def body(u, shuff_idx):
        i_mod = ((i * hvp_batch_size) % x.shape[0]) // hvp_batch_size
        start, end = i_mod * hvp_batch_size, (i_mod + 1) * hvp_batch_size
        x_batch = x[shuff_idx][start:end]
        y_batch = y[shuff_idx][start:end]
        
        batch_hvps = hvp(model, x_batch, y_batch, u)
        new_estimate = [a + (1 - damping) * b - c / scale for a, b, c in zip(v, u, batch_hvps)]
        return new_estimate

    estimate = None
    for r in range(repetitions):
        u = v
        update_min = np.inf
        no_update_iters = 0
        
        for i in range(n_batches):
            new_estimate = body(u, shuffle_indices[r])
            update_norm = sum(torch.norm(old - new).item() for old, new in zip(u, new_estimate))
            
            if early_stopping and update_norm > update_min and no_update_iters >= patience:
                if i < patience:
                    break
                else:
                    return u, True
            
            if update_norm < update_min:
                update_min = update_norm
                no_update_iters = 0
            else:
                no_update_iters += 1
                
            if verbose:
                print(f"Iteration {i}: update norm = {update_norm}")

            if hvp_logger is not None:
                hvp_logger.log(step=hvp_logger.step, inner_step=i, update_norm=update_norm)
            
            if i + 1 == n_batches:
                print(f"No convergence after {i + 1} iterations. Stopping.")
                break

            u = new_estimate

        res_upscaled = [r / scale for r in new_estimate]
        if estimate is None:
            estimate = [r / repetitions for r in res_upscaled]
        else:
            for j in range(len(estimate)):
                estimate[j] += res_upscaled[j] / repetitions

    diverged = not all([torch.isfinite(torch.norm(e)) for e in estimate])
    return estimate, diverged


def get_gradients_diff(model, criterion, x_tensor, y_tensor, x_delta_tensor, y_delta_tensor, batch_size=1024):
    """
    Compute d/dW [ Loss(x_delta, y_delta) - Loss(x,y) ]
    This saves one gradient call compared to calling `get_gradients` twice.
    """
    assert x_tensor.shape == x_delta_tensor.shape and y_tensor.shape == y_delta_tensor.shape
    grads = []

    model.train()
    for start in range(0, x_tensor.shape[0], batch_size):
        model.zero_grad()
        
        x_batch = x_tensor[start:start + batch_size]
        y_batch = y_tensor[start:start + batch_size]
        x_delta_batch = x_delta_tensor[start:start + batch_size]
        y_delta_batch = y_delta_tensor[start:start + batch_size]
        
        result_x = model(x_batch)
        result_x_delta = model(x_delta_batch)
        
        loss_x = criterion(result_x, y_batch)
        loss_x_delta = criterion(result_x_delta, y_delta_batch)
        
        diff = loss_x_delta - loss_x
        diff.backward()

        grads.append([param.grad.clone() for param in model.parameters() if param.requires_grad])

    grads = list(zip(*grads))
    for i in range(len(grads)):
        grads[i] = torch.stack(grads[i], dim=0).sum(dim=0)
    
    return grads

def approx_retraining(model, criterion, z_x, z_y, z_x_delta, z_y_delta, order=2, hvp_x=None, hvp_y=None, hvp_logger=None,
                      conjugate_gradients=False, verbose=False, **unlearn_kwargs):
    """ Perform parameter update using influence functions. """
    """ z_x, z_y 标签翻转后要遗忘的数据 z_x_delta, z_y_delta 是原始数据"""
    if order == 1:
        tau = unlearn_kwargs.get('tau', 1)

        # first order update
        diff = get_gradients_diff(model, criterion, z_x, z_y, z_x_delta, z_y_delta)
        d_theta = diff
        diverged = False
    elif order == 2:
        tau = 1  # tau not used by second-order

        # second order update
        diff = get_gradients_diff(model, criterion, z_x, z_y, z_x_delta, z_y_delta)
        # skip hvp if diff == 0
        if sum(torch.sum(d) for d in diff) == 0:
            d_theta = diff
            diverged = False
        elif conjugate_gradients:
            raise NotImplementedError('Conjugate Gradients is not implemented yet!')
        else:
            assert hvp_x is not None and hvp_y is not None
            d_theta, diverged = get_inv_hvp_lissa(model, hvp_x, hvp_y, diff, verbose=verbose, hvp_logger=hvp_logger, **unlearn_kwargs)
    
    if order != 0:
        # only update trainable weights
        update_pos = len([p for p in model.parameters() if p.requires_grad]) - len(d_theta)
        theta_approx = [w - tau * d_theta.pop(0) if i >= update_pos else w for i, w in enumerate(model.parameters()) if w.requires_grad]

        for i, param in enumerate(model.parameters()):
            if param.requires_grad:
                param.data = theta_approx.pop(0).data
    
    return [param.data.clone() for param in model.parameters()], diverged
