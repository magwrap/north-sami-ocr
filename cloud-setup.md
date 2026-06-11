# Cloud / remote machine setup

### 1. Clone the repo

Public HTTPS:
```shell
git clone https://github.com/<your-user>/north-sami-ocr.git
```

SSH:
```shell
git clone git@github.com:<your-user>/north-sami-ocr.git
```

Personal Access Token (depricated, the repo is no longer private):
```shell
git clone https://<YOUR_GITHUB_PAT>@github.com/<your-user>/north-sami-ocr.git
```

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
