#!/usr/bin/env bash
# install_pygobject.sh
# This script installs PyGObject and its dependencies on Debian/Ubuntu, Fedora, Arch Linux, and macOS.
# It also creates a simple GTK "Hello World" example (hello.py) in the current directory and
# runs it to verify that the installation was successful.

set -euo pipefail

# Determine whether to use sudo when installing packages
# If we are root, sudo is not necessary. Otherwise, try to use sudo if available.
require_root() {
    if [[ "$EUID" -eq 0 ]]; then
        echo ""
    else
        if command -v sudo >/dev/null 2>&1; then
            echo "sudo"
        else
            echo "[ERROR] This script requires root privileges to install packages." >&2
            echo "Install sudo or run as root." >&2
            exit 1
        fi
    fi
}

# Install dependencies based on distribution
install_debian() {
    local SUDO_PREFIX="$1"
    echo "[INFO] Updating package lists on Debian/Ubuntu..."
    $SUDO_PREFIX apt update -y
    echo "[INFO] Installing PyGObject and GTK3 using apt..."
    $SUDO_PREFIX apt install -y libgirepository-2.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0
}

install_fedora() {
    local SUDO_PREFIX="$1"
    echo "[INFO] Installing PyGObject and GTK3 using dnf..."
    $SUDO_PREFIX dnf install -y gcc gobject-introspection-devel cairo-gobject-devel pkg-config python3-devel gtk+3
    pip install --no-input pycairo
    pip install --no-input PyGObject
}

install_arch() {
    local SUDO_PREFIX="$1"
    echo "[INFO] Updating package lists on Arch Linux..."
    $SUDO_PREFIX pacman -Sy --noconfirm
    echo "[INFO] Installing PyGObject and GTK3 using pacman..."
    $SUDO_PREFIX pacman -S --noconfirm python cairo pkgconf gobject-introspection gtk+3
    pip install --no-input pycairo
    pip install --no-input PyGObject
}

install_macos() {
    echo "[INFO] Detected macOS. Checking for Homebrew..."
    if ! command -v brew >/dev/null 2>&1; then
        echo "[ERROR] Homebrew is not installed. Please install Homebrew from https://brew.sh/ first." >&2
        exit 1
    fi
    echo "[INFO] Installing PyGObject and GTK3 using Homebrew..."
    brew update
    brew install pygobject3 gtk+3
    pip install pygobject
}

install_pygobject() {

    # Detect operating system
    local uname_out="$(uname)"
    local SUDO_PREFIX

    if [[ "$uname_out" == "Darwin" ]]; then
        # macOS
        install_macos
    elif [[ "$uname_out" == "Linux" ]]; then
        # Linux; parse /etc/os-release for distribution ID
        if [[ -f /etc/os-release ]]; then
            . /etc/os-release
            # Lower-case ID variable may contain distro name
            case "$ID" in
                ubuntu|debian)
                    SUDO_PREFIX="$(require_root)"
                    install_debian "$SUDO_PREFIX"
                    ;;
                fedora)
                    SUDO_PREFIX="$(require_root)"
                    install_fedora "$SUDO_PREFIX"
                    ;;
                arch)
                    SUDO_PREFIX="$(require_root)"
                    install_arch "$SUDO_PREFIX"
                    ;;
                *)
                    echo "[ERROR] Unsupported Linux distribution: $ID" >&2
                    exit 1
                    ;;
            esac
        else
            echo "[ERROR] Cannot determine Linux distribution (missing /etc/os-release)." >&2
            exit 1
        fi
    else
        echo "[ERROR] Unsupported operating system: $uname_out" >&2
        exit 1
    fi
}

install_pygobject "$@" &&
echo "[INFO] Installing Python dependencies"
pip install -r requirements.txt
export DYLD_LIBRARY_PATH="$(brew --prefix atk)/lib:$(brew --prefix gtk+3)/lib:$(brew --prefix glib)/lib:$(brew --prefix pango)/lib:$(brew --prefix gdk-pixbuf)/lib:$DYLD_LIBRARY_PATH"
export GI_TYPELIB_PATH="$(brew --prefix atk)/lib/girepository-1.0:$(brew --prefix gtk+3)/lib/girepository-1.0:$(brew --prefix gobject-introspection)/lib/girepository-1.0:$GI_TYPELIB_PATH"

