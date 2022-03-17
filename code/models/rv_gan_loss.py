import torch

"""
https://github.com/SharifAmit/RVGAN/blob/master/src/losses.py
"""


def weighted_feature_matching_loss(y_true, fake_samples, image_input,
                                   real_samples, model, inner_weight, sample_weight):
    y_fake = model([image_input, fake_samples])[1:]
    y_real = model([image_input, real_samples])[1:]

    fm_loss = 0
    for i in range(len(y_fake)):
        if i < 3:
            fm_loss += inner_weight * torch.mean(torch.abs(y_fake[i] - y_real[i]))
        else:
            fm_loss += (1 - inner_weight) * torch.mean(torch.abs(y_fake[i] - y_real[i]))
        fm_loss *= sample_weight
        return fm_loss
