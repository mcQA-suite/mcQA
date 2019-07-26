import os
import setuptools
import mcqa


def read(file):
    return open(os.path.join(os.path.dirname(__file__), file)).read()


setuptools.setup(
    name="mcqa",
    version=mcqa.__version__,
    author="Taycir Yahmed",
    author_email="taycir.yahmed@gmail.com",
    description="Answering multiple choice questions with Language Models.",
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    url="https://github.com/mcQA-suite/mcQA",
    packages=setuptools.find_packages(),
    install_requires=read('requirements.txt').split(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
