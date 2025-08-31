from setuptools import setup, find_packages

setup(
    name="mlselect",
    version="1.0.0",
    description="Machine Learning Algorithm Selection Tool",
    author="ML Select Team",
    author_email="team@mlselect.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "scipy>=1.9.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.10.0",
        "openpyxl>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "mlselect=mlselect:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="machine learning, algorithm selection, data analysis, ml",
    long_description="A command-line tool for automatic machine learning algorithm selection based on data characteristics.",
    long_description_content_type="text/plain",
)