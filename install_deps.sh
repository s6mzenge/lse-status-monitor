#!/bin/bash
# Smart dependency installer for LSE Status Monitor
# Installs core dependencies always, advanced only when needed

set -e

echo "🚀 Smart dependency installation starting..."
START_TIME=$(date +%s)

# Always install core dependencies (lightweight, <3MB total)
echo "📦 Installing core dependencies..."
pip install --upgrade pip wheel setuptools --disable-pip-version-check --timeout 30 --retries 2
pip install -r requirements-core.txt --prefer-binary --disable-pip-version-check --timeout 30 --retries 2

CORE_TIME=$(date +%s)
CORE_DURATION=$((CORE_TIME - START_TIME))
echo "✅ Core dependencies installed in ${CORE_DURATION}s"

# Install advanced dependencies conditionally
INSTALL_ADVANCED=${INSTALL_ADVANCED:-"false"}
MANUAL_RUN=${MANUAL_RUN:-"false"}

if [ "$INSTALL_ADVANCED" = "true" ] || [ "$MANUAL_RUN" != "true" ]; then
    echo "📈 Installing advanced dependencies for enhanced analytics..."
    pip install -r requirements-advanced.txt --prefer-binary --disable-pip-version-check --timeout 60 --retries 2
    
    ADV_TIME=$(date +%s)
    ADV_DURATION=$((ADV_TIME - CORE_TIME))
    echo "✅ Advanced dependencies installed in ${ADV_DURATION}s"
else
    echo "⚡ Skipping advanced dependencies for fast manual run"
fi

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
echo "🎯 Total installation time: ${TOTAL_DURATION}s"

# Verify core functionality
echo "🔍 Verifying installation..."
python -c "
import requests
import bs4
print('✅ Core dependencies working')

# Test optional advanced features
try:
    import numpy, scipy, sklearn, matplotlib
    print('✅ Advanced dependencies available')
except ImportError:
    print('⚡ Advanced dependencies skipped (fast mode)')
"