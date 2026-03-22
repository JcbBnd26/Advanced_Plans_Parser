Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
repoPath = fso.GetParentFolderName(WScript.ScriptFullName)
pythonwPath = fso.BuildPath(repoPath, ".venv\Scripts\pythonw.exe")
command = "\"" & pythonwPath & "\" -m scripts.gui"
shell.CurrentDirectory = repoPath
shell.Run command, 0, False
