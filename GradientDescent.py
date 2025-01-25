import numpy as np

def batch_gradient_descent(w0, loss_func, grad_loss_func, lr=0.01, 
                           loss_stop=1e-4, w_stop=1e-4, max_iter=100 ):
    """
    Function that performs batch_gradient_descent.

    # Arguments
        w0: initial value(s) of the optimisation var(s)
        loss_func: function of w that we seek to minimise
        grad_func: gradient of loss_func wrt w
        lr: learning rate
        loss_stop: stop iterating if the loss changes by less than this (absolute)
        w_stop: stop iterating if w changes by less than this (L2 norm)
        max_iter: stop iterating after iterating this many times

    # Returns
        ws: a list of the w values at each iteration
        losses: a list of the losses at each iteration
    """
    # Create lists that store initial values of loss_function and w
    losses = [ loss_func(w0) ]
    ws = [ w0 ]
    
    # Assume the change in loss and in w is infite initially
    d_loss = np.inf
    d_w = np.inf

    # Loop the following when none of the stopping critierias are met
    iter_count = 0
    while (iter_count <= max_iter) and (d_loss > loss_stop) and (d_w > w_stop):
        # Add the updated w into our list ws
        ws.append(ws[-1] - lr * grad_loss_func(ws[-1]))
        # Add the updated loss into our list losses
        losses.append(loss_func(ws[-1]))
        
        # Update d_loss and d_w with updated w and loss
        d_loss = np.abs(losses[-2] - losses[-1])
        d_w = np.linalg.norm(ws[-2] - ws[-1])

    return ws[1:], losses[1:]


def stochastic_gradient_descent(w0, loss_func, grad_loss_func, data, batch_size=1, lr=0.01, 
                                loss_stop=1e-4, w_stop=1e-4, max_iter=100):
    """
    Function that performs mini-batch gradient descent optimisation.

    # Arguments
        w0: initial value(s) of the optimisation var(s)
        loss_func: function of w that we seek to minimise
        grad_func: gradient of loss_func wrt w
        data: a list of tuples that zip each sample with its corresponding label together
        lr: learning rate
        loss_stop: stop iterating if the loss changes by less than this (absolute)
        w_stop: stop iterating if w changes by less than this (L2 norm)
        max_iter: stop iterating after iterating this many times

    # Returns
        ws: a list of the w values at each iteration
        losses: a list of the losses at each iteration
    """
    # Create lists that store initial values of loss_function and w
    losses = [ loss_func(w0) ]
    ws = [ w0 ]
    
    # Assume the change in loss and in w is infite initially
    d_loss = np.inf
    d_w = np.inf
    
    # Main loop of SGD
    iter_count = 0
    while (iter_count < max_iter) and (d_loss > loss_stop) and (d_w > w_stop):
        # Randomly select a batch of data points for stochastic gradient computation
        batch_indices = np.random.choice(len(data), batch_size, replace=False)
        batch_grad = grad_loss_func(data, ws[-1], batch_indices)  # Compute gradient on batch

        # Update w using stochastic gradient
        new_w = ws[-1] - lr * batch_grad
        ws.append(new_w)
        
        # Compute loss for the new w
        new_loss = loss_func(new_w)
        losses.append(new_loss)

        # Update d_loss and d_w with the updated w and loss
        d_loss = np.abs(losses[-2] - losses[-1])
        d_w = np.linalg.norm(ws[-2] - ws[-1])

        iter_count += 1

    return ws[1:], losses[1:]
    
