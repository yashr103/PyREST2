from setuptools import setup
from Cython.Build import cythonize

# List all the helper files you want to compile.
# DO NOT include main.py here.
files_to_compile = [
    "app.py",
    "analysis.py",
    "dcd_extraction.py",
    "ligand_prep.py",
    "pdb_preprocessing.py",
    "residue_utils.py",
    "simulation_run.py",
    "system_generation.py",
]

setup(ext_modules=cythonize(files_to_compile))
