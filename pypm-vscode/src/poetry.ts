// poetry.ts
import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";
import * as toml from "toml";
import { CONFIG, getPoetryConfigPath } from "./config";

let poetryAvailableCache: { [key: string]: boolean } = {};

export async function isPoetryInstalled(): Promise<boolean> {
  return new Promise((resolve) => {
    const cp = require('child_process');
    cp.exec('poetry --version', (error: any) => {
      resolve(!error);
    });
  });
}

export async function checkPoetryAvailability(folderPath: string): Promise<boolean> {
  if (poetryAvailableCache[folderPath] !== undefined) {
    return poetryAvailableCache[folderPath];
  }

  const filePath = path.join(folderPath, CONFIG.PYPROJECT_TOML);
  if (fs.existsSync(filePath)) {
    try {
      const fileContent = fs.readFileSync(filePath, "utf-8");
      const config = toml.parse(fileContent);
      poetryAvailableCache[folderPath] = !!(config.tool && config.tool.poetry);
      return poetryAvailableCache[folderPath];
    } catch (error) {
      console.error(`Error parsing pyproject.toml: ${error}`);
    }
  }

  poetryAvailableCache[folderPath] = false;
  return false;
}

export async function checkPoetryVirtualEnv(folderPath: string): Promise<boolean> {
  return (
    (await isLocalVenvExists(folderPath)) ||
    (await isGlobalVenvExists(folderPath))
  );
}

async function isLocalVenvExists(folderPath: string): Promise<boolean> {
  const venvPath = path.join(folderPath, CONFIG.VENV_FOLDER);
  return fs.existsSync(venvPath) && fs.lstatSync(venvPath).isDirectory();
}

async function isGlobalVenvExists(folderPath: string): Promise<boolean> {
  const poetryConfigPath = getPoetryConfigPath();

  if (poetryConfigPath && fs.existsSync(poetryConfigPath)) {
    try {
      const configContent = fs.readFileSync(poetryConfigPath, "utf-8");
      const config = toml.parse(configContent);
      const virtualenvsPath = config["virtualenvs.path"];
      if (virtualenvsPath) {
        const projectName = path.basename(folderPath);
        const venvPath = path.join(virtualenvsPath, projectName);
        return fs.existsSync(venvPath) && fs.lstatSync(venvPath).isDirectory();
      }
    } catch (error) {
      console.error(`Error parsing poetry config: ${error}`);
    }
  }

  return false;
}

export function clearPoetryCache(folderPath: string): void {
  delete poetryAvailableCache[folderPath];
}