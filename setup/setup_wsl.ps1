# Enable WSL feature
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# Enable Virtual Machine Platform (required for WSL 2)
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Download and install the WSL2 Linux kernel update package
Invoke-WebRequest -Uri https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi -OutFile "$env:TEMP\wsl_update_x64.msi"
Start-Process msiexec -Wait -ArgumentList '/i', "$env:TEMP\wsl_update_x64.msi", '/quiet', '/norestart'

wsl --update

# Set WSL 2 as the default version
wsl --set-default-version 2

# Install Ubuntu (or any other distribution)
wsl --install -d Ubuntu

# Restart the computer to complete the installation
Restart-Computer
