set -e -x
python main.py --config ./config.yaml
# wait for 3 min and shut down
sleep 180
sudo shutdown -h now
