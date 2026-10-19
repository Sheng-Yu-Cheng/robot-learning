from setuptools import find_namespace_packages, setup

setup(
    name="robot-learning",
    version="0.1.0",
    description="An executable knowledge base for robot learning",
    python_requires=">=3.11",
    packages=find_namespace_packages(
        include=[
            "benchmark", "benchmark.*", "benchmarks", "benchmarks.*",
            "methods", "methods.*", "models", "models.*",
            "paradigms", "paradigms.*", "datasets", "datasets.*",
            "experiments", "experiments.*",
        ],
        exclude=[
            "benchmark.maniskill.maniskill", "benchmark.maniskill.maniskill.*",
            "*.outputs", "*.outputs.*", "*.__pycache__", "*.__pycache__.*",
        ],
    ),
)
