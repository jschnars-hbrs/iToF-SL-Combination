"""Parameter overrides from the GUI: module constants replaced by the JSON file named in $ITOF_OVERRIDES.

The file maps module names to {CONSTANT: value}. Without the variable nothing happens, so notebooks and CLIs
see the values written in the code.
"""

import json
import os

ENV_VAR = "ITOF_OVERRIDES"


def apply(namespace):
    """Replace the constants of the calling module (pass globals()) that the overrides file sets."""
    path = os.environ.get(ENV_VAR)
    if not path:
        return
    with open(path) as f:
        values = json.load(f).get(namespace["__name__"].rsplit(".", 1)[-1], {})
    for name, value in values.items():
        if name not in namespace:
            print(f"(override {name} ignored: not defined in {namespace['__name__']})")
            continue
        namespace[name] = tuple(value) if isinstance(namespace[name], tuple) else value
        print(f"(override {namespace['__name__']}.{name} = {value!r})")
