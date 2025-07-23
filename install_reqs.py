import subprocess
import sys


def install_package(pkg):
    try:
        print(f"Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        print(f"Successfully installed: {pkg}\n")
    except subprocess.CalledProcessError:
        print(f"Failed to install: {pkg}\n")


def read_requirements_and_install(file_path="server_requirements.txt"):
    with open(file_path, "r") as f:
        for line in f:
            pkg = line.strip()
            if pkg and not pkg.startswith("#"):
                install_package(pkg)


if __name__ == "__main__":
    read_requirements_and_install()
