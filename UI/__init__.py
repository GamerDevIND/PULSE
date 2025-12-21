import os
import importlib

exceptions = []

modules = [f[:-3] for f in os.listdir(os.path.dirname(__file__)) 
           if f.endswith('.py') and f != '__init__.py' and f not in exceptions]

for module in modules:
    importlib.import_module(f".{module}", package=__name__)

__all__ = modules  # type: ignore
