from torch.optim import Adam
import torch


def weighted_bce(predictions, targets, weight=1):
    device = predictions.device
    assert device == targets.device
    loss_tensor = - (targets*torch.log(predictions) +
                     (torch.ones(targets.shape, device=device) - targets) *
                     torch.log(torch.ones(predictions.shape, device=device) - predictions))
    weights = torch.ones(loss_tensor.shape, device=device) + \
        (weight * targets - targets)
    weighted_loss_tensor = weights * loss_tensor
    return weighted_loss_tensor.mean()


def adam_discriminative_lr(model, learning_rate, weight_decay, lr_decay):
    layers_with_parameters = []
    for module in model.modules():
        for child in module.children():
            if (list(child.children()) == [] and
                list(child.parameters()) != [] and
                    child.__class__.__name__ != "LayerNorm"):
                layers_with_parameters.append(child)

    params = [
        {"params": layer.parameters(), "lr": learning_rate * lr_decay**index}
        for index, layer in enumerate(reversed(layers_with_parameters))
    ]

    return Adam(params, lr=learning_rate, weight_decay=weight_decay)
