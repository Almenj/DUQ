try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

with open("README.md", 'r') as f:
    long_description = f.read()

requirements = [
    "arviz",
    "Flask",
    "livelossplot",
    "matplotlib",
    "jupyter",
    "joypy",
    "numpy",
    "pandas",
    "scikit_learn",
    "scipy",
    "seaborn",
    "torch",
    "wandb",
]

setup(name='duq',
      use_scm_version=True,
      version='1.0',
      setup_requires=['setuptools_scm'],
      license="MIT",
      description="Uncertainty Quantification for Deep Learning",
      long_descritpion=long_description,
      url='https://github.com/ese-msc-2021/irp-aol21',
      author="Archie Luxton",
      install_requires=requirements,
      packages=find_packages(),
      author_email='a.o.luxton@gmail.com')
