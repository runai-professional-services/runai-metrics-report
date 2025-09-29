#!/bin/bash

# Release script for RunAI Metrics Report
# Usage: ./release.sh v0.2.0 "Release description"

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <version> [description]"
    echo "Example: $0 v0.2.0 'Add new metrics features'"
    echo "Example: $0 v0.1.1 'Fix memory leak bug'"
    exit 1
fi

VERSION=$1
DESCRIPTION=${2:-"Release $VERSION"}
CHART_VERSION=${VERSION#v}  # Remove 'v' prefix for chart version

echo "🚀 Creating release $VERSION..."
echo "📋 Description: $DESCRIPTION"
echo ""

# Validate version format
if [[ ! $VERSION =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "❌ Error: Version must be in format vX.Y.Z (e.g., v0.2.0)"
    exit 1
fi

# Check if we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "❌ Error: Must be on main branch to create release"
    echo "Current branch: $CURRENT_BRANCH"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "❌ Error: You have uncommitted changes. Please commit or stash them first."
    git status --porcelain
    exit 1
fi

# Check if tag already exists
if git rev-parse "$VERSION" >/dev/null 2>&1; then
    echo "❌ Error: Tag $VERSION already exists"
    exit 1
fi

# Pull latest changes
echo "📥 Pulling latest changes..."
git pull origin main

# Update Chart.yaml versions
echo "📝 Updating Chart.yaml..."
if ! command -v yq &> /dev/null; then
    echo "❌ Error: yq is required but not installed."
    echo "Install with: brew install yq"
    exit 1
fi

yq eval ".version = \"$CHART_VERSION\"" -i chart/metrics-report/Chart.yaml
yq eval ".appVersion = \"$VERSION\"" -i chart/metrics-report/Chart.yaml

echo "✅ Updated Chart.yaml:"
echo "   version: $CHART_VERSION"
echo "   appVersion: $VERSION"
echo ""

# Show the changes
echo "📋 Changes to be committed:"
git diff chart/metrics-report/Chart.yaml
echo ""

# Commit Chart.yaml changes
echo "💾 Committing Chart.yaml changes..."
git add chart/metrics-report/Chart.yaml
git commit -m "Update Chart.yaml for release $VERSION"

# Create and push tag
echo "🏷️  Creating tag $VERSION..."
git tag -a "$VERSION" -m "$DESCRIPTION"

echo "📤 Pushing changes and tag..."
git push origin main
git push origin "$VERSION"

# Create GitHub release
echo "🎉 Creating GitHub release..."
if ! command -v gh &> /dev/null; then
    echo "⚠️  Warning: GitHub CLI (gh) not found."
    echo "Please install with: brew install gh"
    echo "Or create the release manually at:"
    echo "https://github.com/runai-professional-services/runai-metrics-report/releases/new?tag=$VERSION"
else
    gh release create "$VERSION" \
        --title "Release $VERSION" \
        --notes "$DESCRIPTION

## Installation

\`\`\`bash
helm repo add metrics https://runai-professional-services.github.io/runai-metrics-report
helm repo update
helm install my-metrics metrics/metrics-report --version $CHART_VERSION
\`\`\`

## Docker Image

\`\`\`bash
docker pull bsoper/metrics-report:$VERSION
\`\`\`" \
        --target main

    echo "✅ GitHub release created successfully!"
fi

echo ""
echo "🎯 Release $VERSION completed successfully!"
echo ""
echo "🔗 Monitor the workflow at:"
echo "   https://github.com/runai-professional-services/runai-metrics-report/actions"
echo ""
echo "📦 The workflow will automatically:"
echo "   - Build Docker image: bsoper/metrics-report:$VERSION"
echo "   - Package Helm chart: metrics-report-$CHART_VERSION.tgz"
echo "   - Publish to Helm repository"
echo "   - Attach chart to GitHub release"
echo ""
echo "⏰ Allow 2-3 minutes for the workflow to complete."
