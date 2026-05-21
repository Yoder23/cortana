from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cortana",
    version="1.0.0",
    description="Formally verified AI companion with hard-constraint safety and universe separation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yoder23",
    url="https://github.com/Yoder23/cortana",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21",
    ],
    extras_require={
        "embeddings": ["sentence-transformers>=2.2"],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="ai safety alignment formal-verification universe-separation hallucination-detection",
)
