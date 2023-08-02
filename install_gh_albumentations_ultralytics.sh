if [ $EUID != 0 ]; then
    sudo "$0" "$@"
    exit $?
fi


echo "[Setup] Installing Github CLI"

type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y

echo "[Setup] Installed Github CLI"
echo "[Setup] Installing Ultralytics"
pip install ultralytics
echo "[Setup] Installed Ultralytics"
echo "[Setup] Installing Albumentations"
pip install albumentations
echo "[Setup] Installed Albumentations"