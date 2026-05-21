#!/usr/bin/env python3

import importlib
import platform
import sys

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata


PACKAGES = [
    ("rospy", "rospy"),
    ("cv_bridge", "cv_bridge"),
    ("numpy", "numpy"),
    ("yaml", "PyYAML"),
    ("gymnasium", "gymnasium"),
    ("stable_baselines3", "stable-baselines3"),
    ("torch", "torch"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
]


def package_version(module, distribution):
    try:
        return importlib_metadata.version(distribution)
    except Exception:
        return str(getattr(module, "__version__", "unknown"))


def main():
    print("sys.executable={}".format(sys.executable))
    print("python_version={}".format(platform.python_version()))

    missing = []
    for module_name, distribution in PACKAGES:
        try:
            module = importlib.import_module(module_name)
            version = package_version(module, distribution)
            print("{} ok version={}".format(module_name, version))
        except Exception as exc:
            missing.append((module_name, exc))
            print("{} missing error={}".format(module_name, exc))

    if missing:
        print("")
        print("Missing Phase 7 Python dependencies:")
        for module_name, exc in missing:
            print("  {}: {}".format(module_name, exc))
        print(
            "Install into system Python user site, for example: "
            "/usr/bin/python3 -m pip install --user "
            "stable-baselines3==2.3.2 gymnasium==0.29.1 'torch<2.5' pandas tensorboard matplotlib pyyaml"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
