# Run from the local VS Code PowerShell terminal ONLY after inspecting sources.
# Initial visibility is private. To publish publicly, change visibility deliberately.
$ErrorActionPreference = 'Stop'
if (-not (Test-Path 'pyproject.toml')) { throw 'pyproject.toml not found.' }
if (Test-Path '.git') {
    throw 'Git repository exists. Inspect current branch and remote manually; not overwriting.'
}
# Check credentials before writing Git history.
gh auth status
if ($LASTEXITCODE -ne 0) { throw 'GitHub CLI authentication is required.' }
# Safe local source control initialization. Inspect staged contents before committing.
git init -b main
git add .
git diff --cached --stat
Write-Host 'Review staged contents now. Use the documented commands to commit and publish.'
Write-Host 'git commit -m "research: initial DICOM edge inference benchmark"'
Write-Host 'gh repo create swapins/DICOM-Edge-Inference-Benchmark --private --source . --remote origin --push'
