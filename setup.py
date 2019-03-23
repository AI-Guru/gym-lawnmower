from setuptools import setup

setup(
    name='gym_lawnmower',
    version='0.0.1',
    install_requires=['gym', "pygame"],
    package_data={"": ["gym_lawnmower/envs/resources/*.png"]}
)
