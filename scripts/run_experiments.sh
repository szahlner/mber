#!/bin/bash
cd "/home/mber/"

echo "######################################################"
echo "Start preparing run_scripts"
echo "copy script files"
# Copy script files
cp "./scripts/run_sac.sh" "./run_sac.sh"
cp "./scripts/run_ddpg.sh" "./run_ddpg.sh"

echo "set permissions"
# Set permissions
chmod +x "./run_sac.sh"
chmod +x "./run_ddpg.sh"

echo "ALL DONE, you are now prepared to run the experiments!"
echo "######################################################"