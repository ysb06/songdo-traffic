// config.ts
import * as vscode from 'vscode';
import * as path from 'path';

export const CONFIG = {
  PYPROJECT_TOML: "pyproject.toml",
  VENV_FOLDER: ".venv",
};

export function getEnvActivationSetting(): boolean {
  const configuration = vscode.workspace.getConfiguration("python");
  return configuration.get<boolean>("terminal.activateEnvironment") ?? false;
}

export function getPoetryConfigPath(): string | null {
  const homeDir = process.env.HOME || process.env.USERPROFILE || "";
  const platform = process.platform;

  if (platform === "win32") {
    return path.join(homeDir, "AppData", "Roaming", "pypoetry", "config.toml");
  } else if (platform === "darwin") {
    return path.join(homeDir, "Library", "Application Support", "pypoetry", "config.toml");
  } else {
    return path.join(homeDir, ".config", "pypoetry", "config.toml");
  }
}