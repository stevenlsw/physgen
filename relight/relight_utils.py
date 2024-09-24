import numpy as np
import torch
import cv2
from torch.optim import Adam

def invert(x):
    out = 1.0 / (x + 1.0)
    return out


def uninvert(x, eps=0.001, clip=True):
    if clip:
        x = x.clip(eps, 1.0)

    out = (1.0 / x) - 1.0
    return out


def spherical2cart(r, theta, phi):
    return [
         r * torch.sin(theta) * torch.cos(phi),
         r * torch.sin(theta) * torch.sin(phi),
         r * torch.cos(theta)
    ]


def run_optimization(params, A, b):
    
    optim = Adam([params], lr=0.01)
    prev_loss = 1000
    
    for i in range(500):
        optim.zero_grad()

        x, y, z = spherical2cart(params[2], params[0], params[1])

        dir_shd = (A[:, 0] * x) + (A[:, 1] * y) + (A[:, 2] * z)
        pred_shd = dir_shd + params[3]

        loss = torch.nn.functional.mse_loss(pred_shd.reshape(-1), b)

        loss.backward()

        optim.step()

        # theta can range from 0 -> pi/2 (0 to 90 degrees)
        # phi can range from 0 -> 2pi (0 to 360 degrees)
        with torch.no_grad():
            if params[0] < 0:
                params[0] = 0
                
            if params[0] > np.pi / 2:
                params[0] = np.pi / 2
                
            if params[1] < 0:
                params[1] = 0
                
            if params[1] > 2 * np.pi:
                params[1] = 2 * np.pi   
                
            if params[2] < 0:
                params[2] = 0
                
            if params[3] < 0.1:
                params[3] = 0.1
        
        delta = prev_loss - loss
            
        if delta < 0.0001:
            break
            
        prev_loss = loss
        
    return loss, params


def test_init(params, A, b):
    x, y, z = spherical2cart(params[2], params[0], params[1])

    dir_shd = (A[:, 0] * x) + (A[:, 1] * y) + (A[:, 2] * z)
    pred_shd = dir_shd + params[3]

    loss = torch.nn.functional.mse_loss(pred_shd.reshape(-1), b)
    return loss


def get_light_coeffs(shd, nrm, img, mask=None):
    valid = (img.mean(-1) > 0.05) * (img.mean(-1) < 0.95)

    if mask is not None:
        valid *= (mask == 0)
    
    nrm = (nrm * 2.0) - 1.0
    
    A = nrm[valid == 1]
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    b = shd[valid == 1]
    
    # parameters are theta, phi, and bias (c)
    A = torch.from_numpy(A)
    b = torch.from_numpy(b)
    
    min_init = 1000
    for t in np.arange(0, np.pi/2, 0.1):
        for p in np.arange(0, 2*np.pi, 0.25):
            params = torch.nn.Parameter(torch.tensor([t, p, 1, 0.5]))
            init_loss = test_init(params, A, b)
    
            if init_loss < min_init:
                best_init = params
                min_init = init_loss
                
    loss, params = run_optimization(best_init, A, b)
    
    x, y, z = spherical2cart(params[2], params[0], params[1])

    coeffs = torch.tensor([x, y, z]).reshape(3, 1).detach().numpy()
    coeffs = np.array([x.item(), y.item(), z.item(), params[3].item()])
    return coeffs
    

def generate_shd(nrm, coeffs, msk, bias=True):
    
    nrm = (nrm * 2.0) - 1.0

    A = nrm.reshape(-1, 3)
    A /= np.linalg.norm(A, axis=1, keepdims=True)

    A_fg = nrm[msk == 1]
    A_fg /= np.linalg.norm(A_fg, axis=1, keepdims=True)

    if bias:
        A = np.concatenate((A, np.ones((A.shape[0], 1))), 1)
        A_fg = np.concatenate((A_fg, np.ones((A_fg.shape[0], 1))), 1)
    
    inf_shd = (A_fg @ coeffs)
    inf_shd = inf_shd.clip(0) + 0.2
    return inf_shd


def writing_video(rgb_list, save_path: str, frame_rate: int = 30):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    h, w, _ = rgb_list[0].shape
    out = cv2.VideoWriter(save_path, fourcc, frame_rate, (w, h))

    for img in rgb_list:
        out.write(img)

    out.release()
    return