"""Mirror of real SimpleITK's ``SimpleITK.SimpleITK`` module, where pickles find ``Image``."""

from . import _filters, _image, _resample

for _module in (_image, _resample, _filters):
    for _name in dir(_module):
        if _name.startswith("sitk") or _name[:1].isupper():
            globals()[_name] = getattr(_module, _name)
del _module, _name
