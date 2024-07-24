import * as vscode from "vscode";
import { getEnvActivationSetting } from "./config";
import { isPoetryInstalled, checkPoetryAvailability, checkPoetryVirtualEnv } from "./poetry";
import { getExtensionSetting } from "./settings";

export async function onTerminalCreated(terminal: vscode.Terminal): Promise<void> {
  const terminalOptions: vscode.TerminalOptions = terminal.creationOptions;
  console.log(terminalOptions);

  if (terminalOptions.hideFromUser) {
    return;
  }

  if (!getExtensionSetting('autoActivatePoetryShell')) {
    return;
  }

  const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
  if (!workspaceFolder) {
    return;
  }
  
  const folderPath = workspaceFolder.uri.fsPath;
  
  if (!(await isPoetryInstalled())) {
    vscode.window.showWarningMessage('Poetry is not installed on your system.');
    return;
  }

  if (
    getEnvActivationSetting() ||
    (await checkPoetryAvailability(folderPath)) ||
    (await checkPoetryVirtualEnv(folderPath))
  ) {
    console.log("Debug Session", vscode.debug.activeDebugSession);
    console.log("Debug Console");
    console.log(vscode.debug.activeDebugConsole);
    terminal.sendText("poetry shell");
  }
}