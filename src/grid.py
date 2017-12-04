import numpy as np

def reflect_padding(X, padding):
    assert len(X.shape) == 3, "Image must be tensor 3D"

    size1 = X.shape[1]
    size2 = X.shape[2]

    new_size1 = size1 + 2*padding
    new_size2 = size2 + 2*padding

    X_new = np.zeros((X.shape[0], new_size1, new_size2))

    Xh = X[:,::-1,:]
    Xv = X[:,:, ::-1]
    Xhv = Xv[:,::-1, :]

    X_new[:, :padding, :padding] = Xhv[:, size1-padding:, size2-padding:]
    X_new[:, :padding, padding:new_size2-padding] = Xh[:, size1 - padding:, :]
    X_new[:, :padding, new_size2-padding:] = Xhv[:, size1 - padding:, :padding]

    X_new[:, padding:new_size1-padding, :padding] = Xv[:, :, size2-padding:]
    X_new[:, padding:new_size1-padding, padding:new_size2-padding] = X[:, :,:]
    X_new[:, padding:new_size1-padding, new_size2-padding:] = Xv[:, :, :padding]

    X_new[:, new_size1-padding:, :padding] = Xhv[:, :padding, size2-padding:]
    X_new[:, new_size1-padding:, padding:new_size2-padding] = Xh[:, :padding, :]
    X_new[:, new_size1-padding:, new_size2-padding:] = Xhv[:, :padding, :padding]

    return X_new

def central_crop(X, crop):
    assert len(X.shape) == 3, "Image must be tensor 3D"
    return X[:, crop:X.shape[1]-crop, crop:X.shape[2]-crop]

def eval_grid(image_size, tile_size):
    assert tile_size > 0, "tile_size must be positive"
    assert image_size >= tile_size, "Image_size must be greater than tile_size"
    if image_size % tile_size == 0:
        n = image_size / tile_size
        x = 0
        q = 0

    else:
        n = image_size // tile_size + 1
        x = (n*tile_size - image_size) // (n-1)
        q = (n*tile_size - image_size) % (n-1)

    p = n - 1 - q

    points = [0]
    for i in range(p):
        points.append(points[-1] + tile_size - x)

    for i in range(q):
        points.append(points[-1] + tile_size - x - 1)

    return points
