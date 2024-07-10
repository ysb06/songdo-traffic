import platform
import subprocess


def install():
    os_name = platform.system()

    subprocess.run(
        ["poetry", "run", "pip", "install", "--upgrade", "pip", "setuptools"],
        check=True,
    )

    subprocess.run(["poetry", "install"], check=True)
    if os_name == "Darwin":  # macOS
        subprocess.run(
            [
                "poetry",
                "run",
                "pip",
                "install",
                "dgl",
                "-f",
                "https://data.dgl.ai/wheels/repo.html",
            ],
            check=True,
        )
    else:
        raise NotImplementedError(f"Unsupported OS: {os_name}")


if __name__ == "__main__":
    install()
