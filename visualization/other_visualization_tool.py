import open3d as o3d
import os
import numpy as np



# ---------------------------------------------------------------------------- #
#                function to iteratively visualize a point cloud               #
# ---------------------------------------------------------------------------- #
def visualize_single_image():
    while 1:
        input_path = input("cloud point path : ")
        if input_path == "q" :
            exit(1)
        pcd = o3d.io.read_point_cloud(input_path)
        xyz = np.array(pcd.points)
        X = xyz[:,0]
        Y = xyz[:,1]
        Z = xyz[:,2]

        print("X max {:.2f} X min {:.2f} X gap {:.2f}".format(X.max(), X.min(), X.max()-X.min()))
        print("Y max {:.2f} Y min {:.2f} Y gap {:.2f}".format(Y.max(), Y.min(), Y.max()-Y.min()))
        print("Z max {:.2f} Z min {:.2f} Z gap {:.2f}".format(Z.max(), Z.min(), Z.max()-Z.min()))

        o3d.visualization.draw_geometries([pcd])
        
        


if __name__ == "__main__":
    
    visualize_single_image()
    