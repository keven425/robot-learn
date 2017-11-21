# import ikpy
# import numpy as np
# from ikpy import plot_utils
#
# my_chain = ikpy.chain.Chain.from_urdf_file("../urdf/abb.urdf")
# target_vector = [-.1, .1, 0.]
# target_frame = np.eye(4)
# target_frame[:3, 3] = target_vector
# init_angles = np.zeros([7])
# angles = my_chain.inverse_kinematics(target_frame, initial_position=init_angles) # skip dummy link
# # radians = np.deg2rad(angles)
# radians = angles
# print("The angles of each joints are : ")
# print(', '.join([str(rad) for rad in radians[1:]]))
#
# real_frame = my_chain.forward_kinematics(angles)
# print("Computed position vector : %s, original position vector : %s" % (real_frame[:3, 3], target_frame[:3, 3]))
#
# # import matplotlib.pyplot as plt
# # ax = plot_utils.init_3d_figure()
# # my_chain.plot(angles, ax, target=target_vector)
# # plt.xlim(-0.2, 0.2)
# # plt.ylim(-0.2, 0.2)
#
# pass
#


import pybullet as p

p.connect(p.DIRECT)
# clid = p.connect(p.SHARED_MEMORY)
# if (clid<0):
# 	p.connect(p.GUI)
id = p.loadURDF("../urdf/abb.urdf", basePosition=[0, 0, 0])
p.resetBasePositionAndOrientation(id, [0, 0, 0], [0, 0, 0, 1])
p.setGravity(0,0,0)
p.setRealTimeSimulation(0)


endeff_i = 6
numJoints = p.getNumJoints(id)

# lower limits for null space
ll = [0., -2.094, -1.48353, 0.785, -2.094, -2.094, -2.094]
# upper limits for null space
ul = [0., 2.094, 1.48353, 2.356, 2.094, 2.094, 2.094]
# joint ranges for null space
jr = [0., 4, 5.8, 4, 5.8, 4, 6]
# restposes for null space
rp = [0, 0, 0, 0, 0, 0, 0]
# joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

for i in range(numJoints):
  p.resetJointState(id, i, rp[i])

useNullSpace = 0
ikSolver = 0

# set target position, orientation
pos = [1, 0., 0.]
orn = p.getQuaternionFromEuler([0, 0, 0])

if (useNullSpace == 1):
  jointPoses = p.calculateInverseKinematics(id, endeff_i, pos, orn, ll, ul, jr, rp)
else:
  jointPoses = p.calculateInverseKinematics(id, endeff_i, pos, orn, jointDamping=jd, solver=ikSolver)

print(jointPoses)
# while (True):
#   i = 1
#   position = 1.
#   p.setJointMotorControl2(bodyIndex=id, jointIndex=i, controlMode=p.POSITION_CONTROL, targetPosition=position,
#                           targetVelocity=0, force=1, positionGain=0.03, velocityGain=1)
#   p.stepSimulation()