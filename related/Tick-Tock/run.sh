python main.py --config ./config.yaml > output.log 2>&1
# wait for 3 min and shut down
sleep 180
sudo shutdown -h now
