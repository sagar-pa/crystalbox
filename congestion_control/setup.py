from setuptools import find_packages, setup

setup(
    name="cc_rl",
    packages=[package for package in find_packages() if package.startswith("cc_rl")],
    include_package_data=True,
    python_requires='>=3.9',
    install_requires=[
        "gymnasium>=0.26",
        "numpy",
        "pandas",
        "tqdm",
        "scipy",
    ],
    extras_require={
        "extra": [
            # For training
            "stable-baselines3>=2.0",
            "matplotlib",
            "seaborn",
            "statsmodels",
            # For smart weighting
            "tsfresh",
            "scikit-learn"
        ],
    },
    description="A simulation gym training environment for Congestion Control and CrystalBox",
    author="Sagar Patel",
    url="https://github.com/sagar-pa/crystalbox",
    author_email="sagar.patel@uci.edu",
    license="MIT",
    version=0.15,
)