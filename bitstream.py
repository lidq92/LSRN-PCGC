import struct
import pandas as pd
from pyntcloud import PyntCloud

def write_floats(fd, values, fmt=">{:d}f"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4  # Assuming float is 4 bytes


def read_floats(fd, n, fmt=">{:d}f"):
    sz = struct.calcsize("f")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_ints(fd, values, fmt=">{:d}i"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


def read_ints(fd, n, fmt=">{:d}i"):
    sz = struct.calcsize("i")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]

def read_body(fd):
    return read_bytes(fd, read_ints(fd, 1)[0])


def write_body(fd, out_strings):
    bytes_cnt = 0
    bytes_cnt += write_ints(fd, (len(out_strings),))
    bytes_cnt += write_bytes(fd, out_strings)
    return bytes_cnt

def load_ply(ply_file):
    base_pc = PyntCloud.from_file(ply_file)
    base_points = base_pc.points.values[:,:3].astype(int)    
    return base_points

def save_ply(points, ply_file):
    cloud = PyntCloud(pd.DataFrame(data=points.astype(float), columns=['x', 'y', 'z']))
    cloud.to_file(ply_file, as_text=True)
    return