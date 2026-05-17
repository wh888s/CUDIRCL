class Normalize(object):
    def __init__(self, normalize_type='dicom'):
        if normalize_type == 'dicom':
            self.max = 1500.0
            self.min = -1024.0
        elif normalize_type == 'sino':
            self.max = 6.9
            self.min = -0.1
        else:
            raise ValueError('Unsupported normalize type: {}'.format(normalize_type))

    def __call__(self, image):
        return (image - self.min) / (self.max - self.min)


class DeNormalize(object):
    def __init__(self, normalize_type='dicom'):
        if normalize_type == 'dicom':
            self.max = 1500.0
            self.min = -1024.0
        elif normalize_type == 'sino':
            self.max = 6.9
            self.min = -0.1
        else:
            raise ValueError('Unsupported normalize type: {}'.format(normalize_type))

    def __call__(self, image):
        return image * (self.max - self.min) + self.min
