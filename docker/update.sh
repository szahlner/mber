#!/bin/bash
# Select right directory
cd "/home/"

# Remove existing
rm -r ./mber
rm -r ./shadowhand-gym

# Clone new
git clone https://github.com/szahlner/mber.git
git clone https://github.com/szahlner/shadowhand-gym.git
pip3 install -e shadowhand-gym

# Set permissions and prepare runs
cd "/home/mber/scripts/"
chmod +x "./run_experiments.sh"
./run_experiments.sh