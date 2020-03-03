from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_desc = f.read()

setup(
    name = "blitz-bayesian-torch",
    packages = find_packages(),
    version = "0.1.1",
    description = "A simple and extensible library to create Bayesian Neural Network Layers on PyTorch without trouble and with full integration with nn.Module and nn.Sequential.",
    author = "Pi Esposito",
    url = "https://github.com/piEsposito/blitz-bayesian-deep-learning",
    long_desc = long_desc,
    long_description_content_type = "text/markdown",
    install_requieres = [
                            "torch==1.3.1",
                            "torchvision==0.4.2",
                        ],
    classifiers = [
                    "Development Status :: 3 - Alpha",
                    "Intended Audience :: Developers",
                    "Programming Language :: Python :: 3.7"
                  ]



)