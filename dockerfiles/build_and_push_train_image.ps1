param(
    [string]$ProjectId = $(gcloud config get-value project),
    [string]$Region = "europe-west1",
    [string]$Repo = "sign-ml",
    [string]$ImageName = "train",
    [string]$Tag = "v1"
)

if (-not $ProjectId) {
    Write-Error "ProjectId is not set. Either configure gcloud or pass -ProjectId."
    exit 1
}

$fullImage = "$Region-docker.pkg.dev/$ProjectId/$Repo/$ImageName`:$Tag"

Write-Host "Using project: $ProjectId" -ForegroundColor Cyan
Write-Host "Region: $Region" -ForegroundColor Cyan
Write-Host "Repository: $Repo" -ForegroundColor Cyan
Write-Host "Image: $fullImage" -ForegroundColor Cyan

Write-Host "Creating Artifact Registry repo (if it does not exist)..." -ForegroundColor Yellow
# This will fail harmlessly if the repo already exists
 gcloud artifacts repositories create $Repo `
  --repository-format=docker `
  --location=$Region 2>$null

Write-Host "Configuring Docker auth for $Region-docker.pkg.dev..." -ForegroundColor Yellow
 gcloud auth configure-docker "$Region-docker.pkg.dev"

Write-Host "Building Docker image..." -ForegroundColor Yellow
 docker build -f dockerfiles/train.dockerfile -t $fullImage .

Write-Host "Pushing Docker image..." -ForegroundColor Yellow
 docker push $fullImage

Write-Host "Done. Pushed image: $fullImage" -ForegroundColor Green
