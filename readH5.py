import h5py




def read_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        
        # Posizione per la traiettoria
        pos = f['processed/com/pos'][:]
        pos_offset = f['processed/com/pos_offset'][:]
        pos += pos_offset # Traiettoria 3D reale
        
        # Tempo e velocità
        com_time = f['processed/com/timestamp'][:]
        vel_3d = f['processed/com/vel'][:]
        radius_real = f['processed/com/turn_radius'][:]
        
        # Accelerazione IMU
        acc_3d = f['sensor/imu/acc'][:]
        imu_time = f['sensor/imu/timestamp'][:]

    print("Posizione (3D reale):", pos)
    print("Tempo (COM):", com_time)
    print("Velocità (3D):", vel_3d)
    print("Raggio di curvatura reale:", radius_real)
    print("Accelerazione (3D):", acc_3d)
    print("Tempo (IMU):", imu_time)


# Example usagee
if __name__ == "__main__":
    file_path = './data/rawH5/F_tr1_d1.h5'  # Replace with your HDF5 file path
    read_h5_file(file_path)
