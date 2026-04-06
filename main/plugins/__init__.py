import os
import pkgutil

exceptions = []

current_dir = os.path.dirname(os.path.abspath(__file__))

exceptions = ['__init__.py']
exc_path = os.path.join(current_dir, 'exceptions.txt')
if os.path.exists(exc_path):
    with open(exc_path, 'r') as f:
        exceptions.extend([ff.strip() for ff in f.readlines()])

a = [m.name for m in pkgutil.iter_modules(__path__) if m.name not in exceptions]
__all__ = a # type: ignore