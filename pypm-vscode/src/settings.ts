// settings.ts
import * as vscode from 'vscode';

export function getExtensionSetting(key: string): any {
    return vscode.workspace.getConfiguration("poetry-vscode").get(key);
}

// 이 함수는 더 이상 필요하지 않으므로 삭제합니다.
// export function setExtensionSetting(key: string, value: any): Thenable<void> {
//     return vscode.workspace.getConfiguration("poetry-vscode").update(key, value, vscode.ConfigurationTarget.Global);
// }