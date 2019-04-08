from setuptools import setup, find_packages

setup(
    name='gym_lawnmower',
    version='0.0.1',
    author="Dr. Tristan Behrens (AI Guru)",
    author_email="tristan@ai-guru.de",
    description="A lawnmoving environment for Deep Reinforcement Learning.",
    long_description="This environment provided you with the opportunity to create an AI that mows the lawn with the best strategy.",
    long_description_content_type="text/markdown",
    url="https://github.com/AI-Guru/gym-lawnmower",
    install_requires=['gym', "pygame"],
    packages=find_packages(),
    package_data={"gym_lawnmower": ["envs/resources/*.png"]}
)
