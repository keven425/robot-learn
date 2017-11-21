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


