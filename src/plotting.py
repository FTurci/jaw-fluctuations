import matplotlib.pyplot as plt


def trisum(image3d):
    fig, ax = plt.subplots(1,3, figsize=(12,6),sharey=True)
    
    ax[0].matshow(image3d.sum(axis=0))
    ax[1].matshow(image3d.sum(axis=1))
    ax[2].matshow(image3d.sum(axis=2))

def trimax(image3d,colorbar=False):
    fig, ax = plt.subplots(1,3, figsize=(12,6),sharey=True)
    
    im0 = ax[0].matshow(image3d.max(axis=0))
    im1 = ax[1].matshow(image3d.max(axis=1))
    im2 = ax[2].matshow(image3d.max(axis=2))
    if colorbar:
      fig.colorbar(im0, ax=ax[0])

