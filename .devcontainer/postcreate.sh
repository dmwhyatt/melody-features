# Disable inaccurate safe repository warning
git config --global --add safe.directory '*'

# Remove unnecessary Git LFS config file
rm -f .git/hooks/pre-push
