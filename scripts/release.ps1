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

if (-not (git diff --quiet; $?) -or -not (git diff --cached --quiet; $?)) {
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
    $args = @($Remote, $tag)
    if ($Force) { $args = @('--force') + $args }
    git push @args
}

Invoke-Step "Creating GitHub release $tag" {
    $args = @('release', 'create', $tag, '--title', $tag, '--notes', $Notes)
    gh @args
}

Write-Host ""
Write-Host "Released $tag" -ForegroundColor Green
gh release view $tag
