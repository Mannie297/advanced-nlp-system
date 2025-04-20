from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nlp_system",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced NLP System for text analysis and processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nlp_system",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "nlp_system=nlp_system.cli:main",
        ],
    },
) 