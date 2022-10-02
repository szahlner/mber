#!/bin/bash
cd "/home/mber/"

echo "######################################################"
echo "Start preparing run_scripts"
echo "copy script files"
# Copy script files
cp "./scripts/run_sac.sh" "./run_sac.sh"
cp "./scripts/run_ddpg.sh" "./run_ddpg.sh"

cp "./scripts/run_sac_ant.sh" "./run_sac_ant.sh"
cp "./scripts/run_sac_ant_per.sh" "./run_sac_ant_per.sh"
cp "./scripts/run_sac_ant_nmer.sh" "./run_sac_ant_nmer.sh"

cp "./scripts/run_sac_halfcheetah.sh" "./run_sac_halfcheetah.sh"
cp "./scripts/run_sac_halfcheetah_per.sh" "./run_sac_halfcheetah_per.sh"
cp "./scripts/run_sac_halfcheetah_nmer.sh" "./run_sac_halfcheetah_nmer.sh"

cp "./scripts/run_sac_hopper.sh" "./run_sac_hopper.sh"
cp "./scripts/run_sac_hopper_per.sh" "./run_sac_hopper_per.sh"
cp "./scripts/run_sac_hopper_nmer.sh" "./run_sac_hopper_nmer.sh"
cp "./scripts/run_sac_hopper_slapp.sh" "./run_sac_hopper_slapp.sh"

cp "./scripts/run_sac_inverted_pendulum.sh" "./run_sac_inverted_pendulum.sh"
cp "./scripts/run_sac_inverted_pendulum_nmer.sh" "./run_sac_inverted_pendulum_nmer.sh"
cp "./scripts/run_sac_inverted_pendulum_per.sh" "./run_sac_inverted_pendulum_per.sh"
cp "./scripts/run_sac_inverted_pendulum_per_nmer.sh" "./run_sac_inverted_pendulum_per_nmer.sh"
cp "./scripts/run_sac_inverted_pendulum_slapp.sh" "./run_sac_inverted_pendulum_slapp.sh"

cp "./scripts/run_sac_walker2d.sh" "./run_sac_walker2d.sh"
cp "./scripts/run_sac_walker2d_nmer.sh" "./run_sac_walker2d_nmer.sh"
cp "./scripts/run_sac_walker2d_per.sh" "./run_sac_walker2d_per.sh"
cp "./scripts/run_sac_walker2d_slapp.sh" "./run_sac_walker2d_slapp.sh"

cp "./scripts/run_sac_fetchreach.sh" "./run_sac_fetchreach.sh"
cp "./scripts/run_sac_fetchpush.sh" "./run_sac_fetchpush.sh"
cp "./scripts/run_sac_handreach.sh" "./run_sac_handreach.sh"

# cp "./scripts/run_ddpg_ant.sh" "./run_ddpg_ant.sh"
# cp "./scripts/run_ddpg_halfcheetah.sh" "./run_ddpg_halfcheetah.sh"
# cp "./scripts/run_ddpg_hopper.sh" "./run_ddpg_hopper.sh"
# cp "./scripts/run_ddpg_hopper_nmer.sh" "./run_ddpg_hopper_nmer.sh"
# cp "./scripts/run_ddpg_inverted_pendulum.sh" "./run_ddpg_inverted_pendulum.sh"
# cp "./scripts/run_ddpg_inverted_pendulum_nmer.sh" "./run_ddpg_inverted_pendulum_nmer.sh"
# cp "./scripts/run_ddpg_inverted_pendulum_per.sh" "./run_ddpg_inverted_pendulum_per.sh"
# cp "./scripts/run_ddpg_inverted_pendulum_per_nmer.sh" "./run_ddpg_inverted_pendulum_per_nmer.sh"
# cp "./scripts/run_ddpg_walker2d.sh" "./run_ddpg_walker2d.sh"

echo "set permissions"
# Set permissions
chmod +x "./run_sac.sh"
# chmod +x "./run_ddpg.sh"

chmod +x "./run_sac_ant.sh"
chmod +x "./run_sac_ant_per.sh"
chmod +x "./run_sac_ant_nmer.sh"

chmod +x "./run_sac_halfcheetah.sh"
chmod +x "./run_sac_halfcheetah_per.sh"
chmod +x "./run_sac_halfcheetah_nmer.sh"

chmod +x "./run_sac_hopper.sh"
chmod +x "./run_sac_hopper_per.sh"
chmod +x "./run_sac_hopper_nmer.sh"
chmod +x "./run_sac_hopper_slapp.sh"

chmod +x "./run_sac_inverted_pendulum.sh"
chmod +x "./run_sac_inverted_pendulum_nmer.sh"
chmod +x "./run_sac_inverted_pendulum_per.sh"
chmod +x "./run_sac_inverted_pendulum_per_nmer.sh"
chmod +x "./run_sac_inverted_pendulum_slapp.sh"

chmod +x "./run_sac_walker2d.sh"
chmod +x "./run_sac_walker2d_nmer.sh"
chmod +x "./run_sac_walker2d_per.sh"
chmod +x "./run_sac_walker2d_slapp.sh"

chmod +x "./run_sac_fetchreach.sh"
chmod +x "./run_sac_fetchpush.sh"
chmod +x "./run_sac_handreach.sh"

# chmod +x "./run_ddpg_ant.sh"
# chmod +x "./run_ddpg_halfcheetah.sh"
# chmod +x "./run_ddpg_hopper.sh"
# chmod +x "./run_ddpg_hopper_nmer.sh"
# chmod +x "./run_ddpg_inverted_pendulum.sh"
# chmod +x "./run_ddpg_inverted_pendulum_nmer.sh"
# chmod +x "./run_ddpg_walker2d.sh"

echo "ALL DONE, you are now prepared to run the experiments!"
echo "######################################################"