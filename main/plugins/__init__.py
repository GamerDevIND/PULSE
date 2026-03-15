import os
import pkgutil

exceptions = []

current_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_dir, 'exceptions.txt'), 'r') as f:
    exceptions = [ff.strip() for ff in f.readlines()]
    if not '__init__.py' in exceptions: exceptions.append('__init__.py')

a = [m.name for m in pkgutil.iter_modules(__path__) if m.name not in exceptions]
__all__ = a # type: ignore