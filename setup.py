from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="planquery",
    version="0.1.0",
    author="PlanQuery Team",
    author_email="contact@planquery.com",
    description="Smart Document Assistant for Architectural Plans",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/planquery/planquery",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
        "all": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "planquery=planquery.cli:cli",
            "planquery-api=planquery.api.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "planquery": [
            "static/*",
            "templates/*",
            "models/*",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/planquery/planquery/issues",
        "Source": "https://github.com/planquery/planquery",
        "Documentation": "https://planquery.readthedocs.io/",
    },
)
