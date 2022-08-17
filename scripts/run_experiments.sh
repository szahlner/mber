#!/bin/bash
cd "/home/mber/"

echo "######################################################"
echo "Start preparing run_scripts"
echo "copy script files"
# Copy script files
cp "./scripts/run_sac.sh" "./run_sac.sh"
cp "./scripts/run_ddpg.sh" "./run_ddpg.sh"
cp "./scripts/run_sac_inverted_pendulum.sh" "./run_sac_inverted_pendulum.sh"
cp "./scripts/run_ddpg_inverted_pendulum.sh" "./run_ddpg_inverted_pendulum.sh"

echo "set permissions"
# Set permissions
chmod +x "./run_sac.sh"
chmod +x "./run_ddpg.sh"
chmod +x "./run_sac_inverted_pendulum.sh"
chmod +x "./run_ddpg_inverted_pendulum.sh"

echo "ALL DONE, you are now prepared to run the experiments!"
echo "######################################################"