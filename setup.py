from setuptools import setup, find_packages

setup(
    name="simple_kmeans",
    version="0.1.1",
    author="Suryansh Shakya",
    author_email="suryanshsinghshakya1@gmail.com",
    description="Implementation of K-means algorithm from scratch with adaptive distance metrics",
    long_description=open("readme.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nullHawk/k-means",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
