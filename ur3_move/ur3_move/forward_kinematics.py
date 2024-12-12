import numpy as np

class UR3ForwardKinematics:
    # Updated DH Parameters based on physical measurements
    DH_PARAMS = [
        [0, 0.1519, np.pi / 2, 0],      # Base to Shoulder
        [0, 0, 0, 0],                   # Shoulder to Elbow
        [-0.24365, 0, 0, 0],            # Elbow to Wrist 1
        [-0.21325, 0.1125, np.pi, 0],   # Wrist 1 to Wrist 2
        [0, 0.08535, -np.pi / 2, 0],    # Wrist 2 to Wrist 3
        [0, 0.0819, 0, 0]               # Wrist 3 to End Effector
    ]

    @staticmethod
    def rotation_matrix_z(theta):
        """Rotation matrix around Z-axis"""
        return np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def translation_matrix(x, y, z):
        """Translation matrix"""
        return np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def rotation_matrix_x(alpha):
        """Rotation matrix around X-axis"""
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha), 0],
            [0, np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 0, 1]
        ])

    @classmethod
    def compute_forward_kinematics(cls, joint_angles):
        """
        Compute end effector position and orientation using forward kinematics
        
        :param joint_angles: List of 6 joint angles in radians
        :return: 4x4 transformation matrix representing end effector pose
        """
        # Start with identity matrix
        T = np.eye(4)
        
        # Compute transformation for each joint
        for i, (angle, params) in enumerate(zip(joint_angles, cls.DH_PARAMS)):
            # Unpack DH parameters
            a, d, alpha, _ = params
            
            # Z-axis rotation by joint angle
            Rz = cls.rotation_matrix_z(angle)
            
            # Translation along Z
            Tz = cls.translation_matrix(0, 0, d)
            
            # Translation along X
            Tx = cls.translation_matrix(a, 0, 0)
            
            # X-axis rotation
            Rx = cls.rotation_matrix_x(alpha)
            
            # Combine transformations
            T = T @ Rz @ Tz @ Tx @ Rx
        
        return T

    @classmethod
    def inverse_kinematics(cls, x_target, y_target, z_target):
        """
        Perform inverse kinematics to calculate joint angles for a given (x, y, z) position of wrist_3_joint.
        
        :param x_target, y_target, z_target: Target position in space for wrist_3_joint
        :return: List of joint angles that should be applied to the robot
        """
        # Define initial joint angles (starting from home position)
        joint_angles = np.zeros(6)
        
        # Define some tolerance for the end-effector position error
        tolerance = 1e-3
        max_iterations = 100
        
        # Starting guess for joint angles
        joint_angles = np.zeros(6)
        
        for _ in range(max_iterations):
            # Compute current end effector position
            T = cls.compute_forward_kinematics(joint_angles)
            
            # Current end-effector position (x, y, z)
            x_current, y_current, z_current = T[0, 3], T[1, 3], T[2, 3]
            
            # Calculate the error between current position and target position
            error = np.array([x_target - x_current, y_target - y_current, z_target - z_current])
            
            # If error is small enough, we have reached the solution
            if np.linalg.norm(error) < tolerance:
                break
            
            # Compute Jacobian matrix using the forward kinematics
            jacobian = np.zeros((3, 6))
            for i in range(6):
                # Perturb each joint and compute the change in position
                joint_angles[i] += tolerance
                T_perturbed = cls.compute_forward_kinematics(joint_angles)
                joint_angles[i] -= tolerance
                
                # Calculate the position change in X, Y, Z due to perturbation
                jacobian[:, i] = [(T_perturbed[0, 3] - x_current) / tolerance,
                                   (T_perturbed[1, 3] - y_current) / tolerance,
                                   (T_perturbed[2, 3] - z_current) / tolerance]
            
            # Solve for joint angle changes (Δθ) using the pseudo-inverse of the Jacobian
            delta_theta = np.linalg.pinv(jacobian) @ error
            
            # Update joint angles
            joint_angles += delta_theta
        
        # Return the computed joint angles
        return joint_angles
