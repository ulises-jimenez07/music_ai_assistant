"""
Code for testing restricted execution with import support
"""

import sys
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from RestrictedPython import compile_restricted
from RestrictedPython.Eval import (
    default_guarded_getitem,
    default_guarded_getiter,
)
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    safe_builtins,
)
from RestrictedPython.PrintCollector import PrintCollector

# Unsafe user code
UNSAFE_CODE = """
try:
    with open('/etc/passwd', 'r') as f:
        print(f.read())
except Exception as e:
    print("Error accessing file:", e)
"""

# Safe user code
SAFE_CODE = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = [1, 2, 3, 4, 5]
mean = np.mean(data)
print("The mean is:")
print(mean)

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print("DataFrame:")
print(df)
print("Sum of column A:")
print(df['A'].sum())
result = printed
"""


def create_restricted_globals():
    # Add plt directly to the globals
    restricted_globals = {
        "_print_": PrintCollector,
        "_getattr_": getattr,
        "_getitem_": default_guarded_getitem,
        "_getiter_": default_guarded_getiter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        # Pre-import modules
        "np": np,
        "pandas": pd,
        "pd": pd,
        "plt": plt,
        "matplotlib": matplotlib,
    }

    # Set up a custom importer
    # pylint: disable=unused-argument
    def custom_importer(name, global_vars=None, local_vars=None, fromlist=(), level_param=0):
        # Direct allowed imports
        if name == "numpy":
            return np
        if name == "pandas":
            return pd
        if name == "matplotlib":
            # Handle "from matplotlib import pyplot" separately
            if fromlist and "pyplot" in fromlist:
                # We've already put plt in the globals
                return matplotlib
            return matplotlib
        if name == "matplotlib.pyplot":
            # Explicitly handle "import matplotlib.pyplot"
            return sys.modules["matplotlib.pyplot"]

        # Block all other imports
        raise ImportError(f"Import of '{name}' is not allowed in restricted environment")

    # Create restricted builtins
    my_builtins = safe_builtins.copy()
    my_builtins["__import__"] = custom_importer
    restricted_globals["__builtins__"] = my_builtins

    return restricted_globals


def execute_restricted_code(code: str):
    # Special preprocessing for matplotlib.pyplot imports
    # Replace direct import with assignment to handle the module name
    modified_code = code.replace("import matplotlib.pyplot as plt", "# matplotlib.pyplot already imported as plt")

    try:
        byte_code = compile_restricted(modified_code, filename="<string>", mode="exec")
        r_globals = create_restricted_globals()
        r_locals: dict[str, Any] = {}

        # pylint: disable=exec-used
        exec(byte_code, r_globals, r_locals)

        return {
            "success": True,
            "output": r_locals.get("_print")(),
        }  # Return the captured 'printed' output

    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Test unsafe code
    print("Trying to execute unsafe code:")
    res = execute_restricted_code(UNSAFE_CODE)
    print(f"Success: {res['success']}")
    if not res["success"]:
        print(f"Error: {res['error']}")
    else:
        print("Output:\n" + str(res["output"]))
    print()

    # Test safe code
    print("Executing safe code:")
    res = execute_restricted_code(SAFE_CODE)
    print(f"Success: {res['success']}")
    if res["success"]:
        print("Output:\n" + str(res["output"]))
    else:
        print(f"Error: {res['error']}")
