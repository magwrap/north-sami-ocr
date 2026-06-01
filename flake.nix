{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};

      # Use Python 3.11 for PyTorch 2.3.1 compatibility
      python = pkgs.python311;

    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          python
          pkgs.mupdf
          pkgs.graphviz
          # Build essentials for compiling Python packages
          pkgs.stdenv.cc.cc.lib
          pkgs.stdenv.cc
          pkgs.zlib
          # Libraries for PyTorch and OpenCV
          pkgs.libGL
          pkgs.glib
          pkgs.libx11
          pkgs.libxext
          pkgs.libxcb
          pkgs.libxkbcommon
        ];

        shellHook = ''
          # Set up library paths for PyTorch and OpenCV
          # Include host CUDA driver path for GPU access
          export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.libGL
            pkgs.glib
            pkgs.libx11
            pkgs.libxext
            pkgs.libxcb
            pkgs.libxkbcommon
          ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

          # Create isolated CUDA driver directory (avoids glibc conflicts)
          CUDA_COMPAT_DIR="$PWD/.nix-cuda-compat"
          if [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
            mkdir -p "$CUDA_COMPAT_DIR"
            for lib in /usr/lib/x86_64-linux-gnu/libcuda*.so*; do
              ln -sf "$lib" "$CUDA_COMPAT_DIR/" 2>/dev/null || true
            done
            # Also link libnvidia-ml for nvidia-smi integration
            for lib in /usr/lib/x86_64-linux-gnu/libnvidia*.so*; do
              ln -sf "$lib" "$CUDA_COMPAT_DIR/" 2>/dev/null || true
            done
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_COMPAT_DIR"
          fi

          # Create and activate venv if it doesn't exist
          if [ ! -d .venv ]; then
            echo "Creating Python virtual environment..."
            ${python}/bin/python -m venv .venv
          fi

          source .venv/bin/activate

          # Install requirements if torch is not installed
          if ! python -c "import torch" 2>/dev/null; then
            echo "Installing Python packages from requirements.txt..."
            pip install --upgrade pip
            pip install -r requirements.txt
          fi

          echo ""
          echo "Development environment ready!"
          echo "Python version: $(python --version)"
          echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Not installed yet')"
        '';
      };
    };
}