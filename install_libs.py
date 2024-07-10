import platform
import subprocess
import torch
import torch.version


def install():
    os_name = platform.system()

    subprocess.run(
        ["poetry", "run", "pip", "install", "--upgrade", "pip", "setuptools"],
        check=True,
    )

    install_command = ["poetry", "run", "pip", "install", "dgl", "-f"]
    if os_name == "Darwin":  # macOS
        install_command.append("https://data.dgl.ai/wheels/repo.html")
    elif os_name == "Windows":
        if torch.cuda.is_available():
            if torch.version.cuda == "12.1":
                install_command.append("https://data.dgl.ai/wheels/cu121/repo.html")
            elif torch.version.cuda == "11.8":
                install_command.append("https://data.dgl.ai/wheels/cu118/repo.html")
            else:
                print(f"CUDA {torch.version.cuda} is not supported by DGL. Installing as CPU version")
                install_command.append("https://data.dgl.ai/wheels/repo.html")
        else:
            install_command.append("https://data.dgl.ai/wheels/repo.html")
    elif os_name == "Linux":
        if torch.cuda.is_available():
            if torch.version.cuda == "12.1":
                install_command.append("https://data.dgl.ai/wheels/cu121/repo.html")
            elif torch.version.cuda == "11.8":
                install_command.append("https://data.dgl.ai/wheels/cu118/repo.html")
            else:
                print(f"CUDA {torch.version.cuda} is not supported by DGL. Installing as CPU version")
                install_command.append("https://data.dgl.ai/wheels/repo.html")
        else:
            install_command.append("https://data.dgl.ai/wheels/repo.html")
    else:
        raise NotImplementedError(f"Unsupported OS: {os_name}")
    
    print("Executing...:", *install_command)
    subprocess.run(install_command, check=True)    


if __name__ == "__main__":
    install()
