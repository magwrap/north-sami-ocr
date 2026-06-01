# Cloud / remote machine setup

### 1. Clone the repo

Public HTTPS (after public release):
```shell
git clone https://github.com/<your-user>/sami-ocr.git
```

SSH (recommended if you have keys set up):
```shell
git clone git@github.com:<your-user>/sami-ocr.git
```

Private repo via Personal Access Token (only if the repo is still private):
```shell
git clone https://<YOUR_GITHUB_PAT>@github.com/<your-user>/sami-ocr.git
```
Replace `<YOUR_GITHUB_PAT>` with a token created at https://github.com/settings/tokens with `repo` scope. Do not commit the URL with the token embedded.

### 2. Install Nix
- https://nixos.wiki/wiki/Nix_Installation_Guide

Add nix to path:
```shell
echo 'source $HOME/.nix-profile/etc/profile.d/nix.sh' >> ~/.bashrc
```

### 3. Start nix development env
```shell
nix develop --extra-experimental-features nix-command --extra-experimental-features flakes
```
