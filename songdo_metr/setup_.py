import platform
import subprocess
import re

def get_installed_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"], universal_newlines=True)
        version_match = re.search(r"release (\d+\.\d+),", output)
        if version_match:
            return version_match.group(1)
        else:
            return None
    except subprocess.CalledProcessError:
        return None
    except FileNotFoundError:
        return None


def install():
    os_name = platform.system()

    # pip 및 setuptools 업그레이드
    subprocess.run(
        ["poetry", "run", "pip", "install", "--upgrade", "pip", "setuptools"],
        check=True,
    )

    # CUDA 설치 여부 및 버전 확인
    cuda_version = get_installed_cuda_version()
    print(f"CUDA Available: {cuda_version is not None}, Version: {cuda_version}")

    torch_install_command = [
        "poetry",
        "run",
        "pip",
        "install",
        "--upgrade",
        "torch==2.2.1",
        "torchvision==0.17.1",
        "torchaudio==2.2.1",
    ]
    if os_name == "Darwin":
        pass
    elif os_name == "Windows" or os_name == "Linux":
        if cuda_version == "12.1":
            torch_install_command.extend(["--index-url", "https://download.pytorch.org/whl/cu121"])
        elif cuda_version == "11.8":
            torch_install_command.extend(["--index-url", "https://download.pytorch.org/whl/cu118"])
    else:
        raise NotImplementedError(f"Unsupported OS: {os_name}")

    # PyTorch 설치
    print("\nInstalling torch as...:", *torch_install_command, end="\n\n")
    subprocess.run(torch_install_command, check=True)

    # DGL 설치 명령어 구성
    dgl_install_command = ["poetry", "run", "pip", "install", "dgl", "-f"]
    if os_name == "Darwin":  # macOS
        dgl_install_command.append("https://data.dgl.ai/wheels/repo.html")
    elif os_name == "Windows":
        if cuda_version == "12.1":
            dgl_install_command.append("https://data.dgl.ai/wheels/cu121/repo.html")
        elif cuda_version == "11.8":
            dgl_install_command.append("https://data.dgl.ai/wheels/cu118/repo.html")
        else:
            dgl_install_command.append("https://data.dgl.ai/wheels/repo.html")
    elif os_name == "Linux":
        if cuda_version == "12.1":
            dgl_install_command.append("https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html")
        elif cuda_version == "11.8":
            dgl_install_command.append("https://data.dgl.ai/wheels/torch-2.2/cu118/repo.html")
        else:
            dgl_install_command.append("https://data.dgl.ai/wheels/repo.html")
    else:
        raise NotImplementedError(f"Unsupported OS: {os_name}")

    # DGL 설치
    print("\nInstalling dgl as...:", *dgl_install_command, end="\n\n")
    subprocess.run(dgl_install_command, check=True)

    print("\nInstallation completed.", end="\n\n")
    import torch
    import dgl
    print("PyTorch Version:", torch.__version__)
    print("DGL Version:", dgl.__version__)


if __name__ == "__main__":
    install()
