// extension.ts
import * as vscode from 'vscode';
import * as path from 'path';
import { CONFIG } from './config';

async function onTerminalCreated(terminal: vscode.Terminal): Promise<void> {
  const terminalOptions: vscode.TerminalOptions = terminal.creationOptions;
  console.log(terminalOptions);
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  const onTerminalCreatedEvent = vscode.window.onDidOpenTerminal(onTerminalCreated);
  const poetryProjectWatcher = vscode.workspace.createFileSystemWatcher(`**/${CONFIG.PYPROJECT_TOML}`);


  poetryProjectWatcher.onDidChange((e) => {
    console.log("On Did Change");
    console.log(e);
  });

  poetryProjectWatcher.onDidCreate((e) => {
    console.log("On Did Create");
    console.log(e);
  });

  poetryProjectWatcher.onDidDelete((e) => {
    console.log("On Did Delete");
    console.log(e);
  });

  context.subscriptions.push(onTerminalCreatedEvent);
  context.subscriptions.push(poetryProjectWatcher);

  vscode.workspace.onDidChangeConfiguration((e) => {
    console.log(e);
  });
}

export function deactivate(): void {
  console.log("Poetry VSCode deactivated");
}