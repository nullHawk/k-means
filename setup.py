from setuptools import setup, find_packages

setup(
    name="simple_kmeans",
    version="0.1",
    author="Suryansh Shakya",
    description="A custom K-Means clustering algorithm with adaptive distance metrics",
    packages=find_packages(),
    install_requires=["numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
