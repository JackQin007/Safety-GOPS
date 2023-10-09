from setuptools import setup, find_packages

setup(
    name='quadrotor',
    packages=[package for package in find_packages() if package.startswith('quadrotor')],
    install_requires=[],
    python_requires='>=3.6',
)
