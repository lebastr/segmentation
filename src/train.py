import numpy as np
from PIL import Image as PILImage

class BatchGenerator(object):
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

    def __call__(self, batch_size=5):
        def make_augmentation():
            if self.random_crop:
                angle = np.random.uniform(-self.rotate_amp, self.rotate_amp)
                bound = 1 + int(self.net_in_size * (1 + np.sin(2 * np.pi * np.abs(angle) / 180)))
                if (bound - self.net_in_size) % 2 != 0:
                    bound += 1

                x_crop = int(np.random.uniform(0, self.xshape[1] - bound))
                y_crop = int(np.random.uniform(0, self.xshape[2] - bound))
                padding = (bound - self.net_in_size) / 2

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
            for i in range(0, len(self.points), batch_size):
                points  = self.points[i:i + batch_size]

                features_batch = []
                target_batch = []

                for p in points:
                    augment = make_augmentation()

                    X = self.get_features(p)
                    Y = self.get_target(p)

                    X = augment(X)
                    Y = augment(Y)
                    Y = crop_target(Y)
                    features_batch.append(X)
                    target_batch.append(Y)

                features_batch = np.array(features_batch)
                target_batch = np.array(target_batch)

                yield features_batch, target_batch