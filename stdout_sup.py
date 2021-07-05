# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:07:36 2021
import and define stdout suppressor:
@author: effi
"""

from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
@contextmanager

def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)