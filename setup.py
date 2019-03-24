from setuptools import setup, find_packages

setup(
    name='gym_lawnmower',
    version='0.0.1',
    author="Dr. Tristan Behrens (AI Guru)",
    author_email="tristan@ai-guru.de",
    description="A small example package",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    install_requires=['gym', "pygame"],
    packages=find_packages(),
    package_data={"gym_lawnmower": ["envs/resources/*.png"]}
)
