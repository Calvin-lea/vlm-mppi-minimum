from setuptools import find_packages, setup


setup(
    name="vlm-mppi-minimum",
    version="0.1.0",
    description="Oracle multi-hypothesis semantic priors for MPPI",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=["numpy>=1.19"],
)

