def get_plane_from_depth(depth):

def output_xyz_gap(pcd):
    xyz = np.array(pcd.points)
    X = xyz[:,0]
    Y = xyz[:,1]
    Z = xyz[:,2]

    print("X max {:.2f} X min {:.2f} X gap {:.2f}".format(X.max(), X.min(), X.max()-X.min()))
    print("Y max {:.2f} Y min {:.2f} Y gap {:.2f}".format(Y.max(), Y.min(), Y.max()-Y.min()))
    print("Z max {:.2f} Z min {:.2f} Z gap {:.2f}".format(Z.max(), Z.min(), Z.max()-Z.min()))