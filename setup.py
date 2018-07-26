import io

from setuptools import setup


def get_requirements(file_name):
    return [i.strip() for i in io.open(file_name).readlines()]


setup(
    name='canary',
    packages=['canary'],
    install_requires=get_requirements('requirements.txt'),
)
