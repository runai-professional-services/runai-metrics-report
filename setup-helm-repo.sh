#!/bin/bash

# Script to initialize GitHub Pages for Helm repository
# Run this once to set up your Helm chart repository

echo "Setting up GitHub Pages branch for Helm repository..."

# Create and switch to gh-pages branch
git checkout --orphan gh-pages

# Remove all files from the new branch
git rm -rf .

# Create initial index.yaml
cat > index.yaml << EOF
apiVersion: v1
entries: {}
generated: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF

# Create a simple README for the Helm repository
cat > README.md << EOF
# RunAI Metrics Report Helm Repository

This is the Helm chart repository for the RunAI Metrics Report.

## Usage

\`\`\`bash
helm repo add metrics https://runai-professional-services.github.io/runai-metrics-report
helm repo update
helm search repo metrics
\`\`\`

## Available Charts

- **metrics-report**: Kubernetes CronJob for generating RunAI metrics reports

EOF

# Commit and push the initial gh-pages branch
git add .
git config --local user.email "action@github.com"
git config --local user.name "GitHub Action"
git commit -m "Initialize Helm repository"
git push origin gh-pages

# Switch back to main branch
git checkout main

echo "✅ GitHub Pages branch created successfully!"
echo ""
echo "Next steps:"
echo "1. Go to GitHub Settings → Pages"
echo "2. Set Source to 'GitHub Actions'"
echo "3. Create a release to test the workflow"
echo ""
echo "Your Helm repository will be available at:"
echo "https://runai-professional-services.github.io/runai-metrics-report"
