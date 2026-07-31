#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidatePattern('^\d+\.\d+\.\d+([-.].+)?$')]
    [string]$Version,

    [Parameter(Mandatory = $true, Position = 1)]
    [string]$Notes,

    [string]$Remote = 'origin',
    [string]$Branch = 'main',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

function Invoke-Step {
    param([string]$Label, [scriptblock]$Action)
    Write-Host "==> $Label" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed ($Label) with exit code $LASTEXITCODE"
    }
}

$tag = "v$Version"

$currentBranch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($currentBranch -ne $Branch) {
    throw "Refusing to release: on branch '$currentBranch', expected '$Branch'. Use -Branch to override."
}

git diff --quiet
$unstagedDirty = ($LASTEXITCODE -ne 0)
git diff --cached --quiet
$stagedDirty = ($LASTEXITCODE -ne 0)
if ($unstagedDirty -or $stagedDirty) {
    throw "Refusing to release: working tree has uncommitted changes."
}

$existingTag = git tag --list $tag
if ($existingTag) {
    if (-not $Force) {
        throw "Tag $tag already exists locally. Delete it or pass -Force to move it."
    }
    Invoke-Step "Deleting existing local tag $tag" { git tag -d $tag | Out-Null }
}

Invoke-Step "Pushing $Branch to $Remote" {
    git push $Remote $Branch
}

Invoke-Step "Creating annotated tag $tag" {
    git tag -a $tag -m "Release $Version - $Notes"
}

Invoke-Step "Pushing tag $tag" {
    $pushArgs = @($Remote, $tag)
    if ($Force) { $pushArgs = @('--force') + $pushArgs }
    git push @pushArgs
}

Invoke-Step "Creating GitHub release $tag" {
    gh release create $tag --title $tag --notes $Notes
}

Write-Host ""
Write-Host "Released $tag" -ForegroundColor Green
gh release view $tag
