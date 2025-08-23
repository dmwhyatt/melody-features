#!/bin/bash
set -e

echo "IDyOM installer for Linux/macOS"

# Detect if running in Docker container (no sudo available)
if [ -f /.dockerenv ] || [ -f /proc/1/cgroup ] && grep -q docker /proc/1/cgroup; then
    echo "Detected Docker container - running without sudo"
    DOCKER_MODE=true
else
    DOCKER_MODE=false
fi

# Detect OS and set package manager
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ "$DOCKER_MODE" = true ]; then
        PM="apt-get"
    else
        PM="sudo apt-get"
    fi
    UPDATE_CMD="$PM update"
    INSTALL_CMD="$PM install -y"
    PKGS="sbcl sqlite3 libsqlite3-dev wget curl unzip bzip2"
elif [[ "$OSTYPE" == "linux-musl"* ]]; then
    # Alpine Linux
    if [ "$DOCKER_MODE" = true ]; then
        PM="apk"
    else
        PM="sudo apk"
    fi
    UPDATE_CMD="$PM update"
    INSTALL_CMD="$PM add"
    PKGS="sbcl sqlite sqlite-dev wget curl unzip bzip2"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if ! command -v brew >/dev/null 2>&1; then
        echo "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    PM="brew"
    UPDATE_CMD="$PM update"
    INSTALL_CMD="$PM install"
    PKGS="sbcl sqlite3 wget curl unzip bzip2"
else
    echo "Unsupported OS: $OSTYPE"
    echo "For Windows, please use WSL or run this script in Git Bash/Cygwin."
    exit 1
fi

echo "Updating package manager..."
eval $UPDATE_CMD

echo "Installing dependencies: $PKGS"
eval $INSTALL_CMD $PKGS

# Install Quicklisp
if [ ! -d "$HOME/quicklisp" ]; then
    mkdir -p "$HOME/quicklisp"
    wget http://beta.quicklisp.org/quicklisp.lisp -O "$HOME/quicklisp/quicklisp.lisp"
    sbcl --load "$HOME/quicklisp/quicklisp.lisp" \
         --eval "(quicklisp-quickstart:install :path \"$HOME/quicklisp/\")" \
         --eval "(ql:add-to-init-file)" \
         --eval "(ql:quickload \"quicklisp-slime-helper\")" \
         --quit
fi

# Install IDyOM source
if [ ! -d "$HOME/quicklisp/local-projects/idyom" ]; then
    mkdir -p "$HOME/quicklisp/local-projects"
    wget https://github.com/mtpearce/idyom/archive/refs/heads/master.zip -O /tmp/idyom.zip
    unzip /tmp/idyom.zip -d "$HOME/quicklisp/local-projects/"
    mv "$HOME/quicklisp/local-projects/idyom-master" "$HOME/quicklisp/local-projects/idyom"
fi

# Create IDyOM data directories
mkdir -p "$HOME/idyom/db" "$HOME/idyom/data/cache" "$HOME/idyom/data/models" "$HOME/idyom/data/resampling"

# Create the .sbclrc file if it doesn't exist and append IDyOM configurations
SBCLRC="$HOME/.sbclrc"
SETUP_LISP_PATH="$HOME/quicklisp/setup.lisp"
LOAD_COMMAND="(load \"$SETUP_LISP_PATH\")" # Use quoted path for robustness
IDYOM_CONFIG_MARKER=";; IDyOM Configuration (v3)"

# Create .sbclrc if it doesn't exist to ensure the file is writeable
touch "$SBCLRC"

# 1. Robustly ensure Quicklisp is loaded. Prepend the load command if it's missing.
if ! grep -q -F -- "$LOAD_COMMAND" "$SBCLRC"; then
    echo "Adding Quicklisp loader to $SBCLRC..."
    # Use a temp file to prepend the command, which is safer and more portable than sed -i
    echo "$LOAD_COMMAND" > "$SBCLRC.tmp"
    cat "$SBCLRC" >> "$SBCLRC.tmp"
    mv "$SBCLRC.tmp" "$SBCLRC"
fi

# 2. Add IDyOM-specific configuration if it's not present
if ! grep -q -F -- "$IDYOM_CONFIG_MARKER" "$SBCLRC"; then
    echo "Adding IDyOM configuration to $SBCLRC..."
    # Append the configuration
    {
        echo ""
        echo "$IDYOM_CONFIG_MARKER"
        echo ";; These are defined at the top level to ensure they exist globally."
        echo "(defvar *idyom-root* \"$HOME/idyom/\")"
        echo "(defvar *idyom-message-detail-level* 1)"
        echo ""
        echo ";; Load CLSQL by default"
        echo "(ql:quickload \"clsql\")"
        echo ""
        echo ";; Helper function to connect to the IDyOM database"
        echo "(defun start-idyom ()"
        echo "   (ql:quickload \"idyom\")"
        echo "   (clsql:connect '(\"$HOME/idyom/db/database.sqlite\") :if-exists :old :database-type :sqlite3))"
    } >> "$SBCLRC"
fi

# Initialize IDyOM and its database
sbcl --non-interactive \
     --load "$HOME/quicklisp/setup.lisp" \
     --eval "(ql:quickload \"clsql\")" \
     --eval "(defun start-idyom () (ql:quickload \"idyom\") (clsql:connect '(\"$HOME/idyom/db/database.sqlite\") :if-exists :old :database-type :sqlite3))" \
     --eval "(start-idyom)" \
     --eval "(idyom-db:initialise-database)" \
     --eval "(quit)"

echo "IDyOM installation complete!"
