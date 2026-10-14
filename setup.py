from setuptools import setup, find_namespace_packages

setup(
    name="robot-learning",
    version="0.1.0",

    packages=find_namespace_packages(
        include=[
            "benchmarks*",
            "methods*",
            "models*",
            "paradigms*",
            "datasets*", 
            "experiments*"
        ]
    )
)
