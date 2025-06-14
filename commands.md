## Record a dataset

python -m lerobot.record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem59700733641 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 1, width: 1920, height: 1080, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record-test \
    --dataset.num_episodes=2 \
    --dataset.single_task="Grab the black cube"

## Calibrate 

python -m lerobot.calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodem59700733641 --robot.id=fol


python -m lerobot.calibrate --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem59700740331 --teleop.id=lea 

Follower arm
/dev/tty.usbmodem59700733641
Leader arm
/dev/tty.usbmodem59700740331