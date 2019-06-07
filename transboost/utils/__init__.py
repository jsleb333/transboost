try:
    from misc import *
    from affine_transform import *
    from comparable_mixin import ComparableMixin
except ModuleNotFoundError:
    from .misc import *
    from .affine_transform import *
    from .comparable_mixin import ComparableMixin
