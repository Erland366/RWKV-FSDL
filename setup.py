from setuptools import find_packages, setup

setup(
    name="rwkv_fsdl",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "jaxtyping",
        "torch",
    ],
    entry_points={"console_scripts": ["rwkv = rwkv_fsdl.rwkv:main"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
