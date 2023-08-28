from setuptools import setup, find_packages

requirements = [
    "torch>=2.0",
    "numpy",
    "matplotlib",
    "qiskit",
    "pylatexenc"
]

setup(
    name='deepquantum',
    version='0.0.4',
    packages=find_packages(where="."),
    url='',
    license='',
    author='TuringQ',
    package_data={'': ['*.so', '*.pyd']},
    include_package_data=True,
    install_requires=requirements,
    description='DeepQuantum for quantum computing'
)