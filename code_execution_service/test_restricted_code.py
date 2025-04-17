"""
Code for testing restricted execution
"""

from typing import Any

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
    return {
        "__builtins__": safe_builtins.copy(),
        "_print_": PrintCollector,
        "_getattr_": getattr,
        "_getitem_": default_guarded_getitem,
        "_getiter_": default_guarded_getiter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "pd": pd,
        "np": np,
    }


def execute_restricted_code(code: str):
    try:
        byte_code = compile_restricted(code, filename="<string>", mode="exec")
        r_globals = create_restricted_globals()
        r_locals: dict[str, Any] = {}
        # pylint: disable=exec-used
        exec(byte_code, r_globals, r_locals)
        # print(r_globals)
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
