#!/usr/bin/env python3
"""
R3Bench: A Benchmark for Evaluating Closed-Loop Capabilities of MLLMs
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="r3bench",
    version="1.0.0",
    author="R3Bench Team",
    author_email="",
    description="A Benchmark for Evaluating Closed-Loop Capabilities of MLLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xiaomoguhz/R3-Bench",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        # Optional: the BAGEL UMM backend (reflection + rectification).
        # flash-attn must match your CUDA/torch. Prefer a prebuilt wheel.
        "bagel": [
            "flash-attn",
            "peft",
            "einops",
            "opencv-python",
            "huggingface_hub",
        ],
    },
    include_package_data=True,
    package_data={
        "r3bench": ["data/*.jsonl"],
    },
)
