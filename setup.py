from setuptools import setup, find_packages
setup(
    name="gridenv",
    version="1.0.0",
    description="GridWorld Survival — Mini-game RL environment (OpenEnv-compliant)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["pydantic>=2.0", "fastapi>=0.111", "uvicorn[standard]>=0.30"],
)
