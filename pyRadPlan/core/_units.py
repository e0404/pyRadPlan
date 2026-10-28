"""The pint unit registry shared by all of pyRadPlan.

Building a registry parses pint's whole unit definition file, which is slow, and
units from different registries cannot be compared or converted into each other.
Every module therefore uses this one registry instead of creating its own.
"""

import pint

ureg = pint.UnitRegistry()
