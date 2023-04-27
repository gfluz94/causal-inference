import setuptools

long_description = """
    This is a library developed to estimate causal effects from observational data.
    The focus is on estimating Average Treatment Effect (ATE) and Conditional Average 
    Treatment Effect (CATE), for known and established frameworks.
"""

setuptools.setup(
    name="causal_inference",
    version="0.0.1",
    author="Gabriel Fernandes Luz",
    author_email="gfluz94@gmail.com",
    description="Package for estimating causal effects from observational data.",
    long_description=long_description,
    packages=list(
        filter(
            lambda x: x.startswith("causal_inference"),
            setuptools.find_packages(),
        )
    ),
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        package.strip()
        for package in open("requirements.txt", encoding="utf-8").readlines()
    ],
)
