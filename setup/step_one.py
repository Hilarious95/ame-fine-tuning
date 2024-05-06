import subprocess

def run_command(command):
    """ Helper function to run commands using PowerShell """
    try:
        result = subprocess.run(["powershell", "-Command", command], capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Error:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute command: {e}")

def setup_miniconda_in_wsl():
    """ Execute the bash script within WSL to set up the environment """
    bash_script_path = "/mnt/d/dev/ame-finetune/setup/setup_miniconda.sh"
    run_command(f"wsl bash {bash_script_path}")

if __name__ == "__main__":
    setup_miniconda_in_wsl()
