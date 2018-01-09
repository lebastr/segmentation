import numpy as np
from PIL import Image as PILImage

class Sampler(object):
    def __init__(self, points, get_features, get_target, net_in_size, net_out_size, rotate_amplitude=10, reflect=False, random_crop=True):
        self.points = points
        self.get_features = get_features
        self.get_target = get_target

        assert (net_in_size - net_out_size) % 2 == 0, "net_in_size - net_out_size must be even"

        self.net_in_size = net_in_size
        self.net_out_size = net_out_size
        self.rotate_amp = rotate_amplitude
        self.reflect = reflect
        self.random_crop = random_crop

        self.xshape = get_features(points[0]).shape

    def __call__(self):
        def make_augmentation():
            if self.random_crop:
                angle = np.random.uniform(-self.rotate_amp, self.rotate_amp)
                bound = 1 + int(self.net_in_size * (1 + np.sin(2 * np.pi * np.abs(angle) / 180)))
                if (bound - self.net_in_size) % 2 != 0:
                    bound += 1

                x_crop = int(np.random.uniform(0, self.xshape[1] - bound))
                y_crop = int(np.random.uniform(0, self.xshape[2] - bound))
                padding = (bound - self.net_in_size) // 2

#                print("bound: %d, padding: %d" % (bound, padding))
                def augment_A(X):
                    X = X[:, x_crop:x_crop+bound, y_crop:y_crop+bound]
                    X = np.array([np.array(PILImage.fromarray(x).rotate(angle)) for x in X])
                    X = X[:, padding:padding + self.net_in_size, padding:padding + self.net_in_size]
                    if self.reflect:
                        X = X[:, ::-1, :]

                    return X
            else:

                def augment_A(X):
                    padding1 = (self.xshape[1] - self.net_in_size) // 2
                    padding2 = (self.xshape[2] - self.net_in_size) // 2
                    return X[:, padding1:padding1 + self.net_in_size, padding2:padding2 + self.net_in_size]

            def augment(X):
                X = augment_A(X)
                if self.reflect:
                    X = X[:, ::-1, :]

                return X

            return augment

        def crop_target(Y):
            padding = (self.net_in_size - self.net_out_size) // 2
            return Y[:, padding:padding + self.net_out_size, padding:padding + self.net_out_size]

        while True:
            i = int(np.random.uniform(0, len(self.points)))

            features = self.get_features(self.points[i])
            target = self.get_target(self.points[i])

            augment = make_augmentation()
            features = augment(features)
            target = augment(target)
            target = crop_target(target)

            yield features, target
