import time
from typing import Union, List
from .abstract import cwipc_abstract_filter
from ..util import cwipc_crop, cwipc_wrapper, cwipc_from_points
import cv2
import pyrealsense2

class Crop2Filter(cwipc_abstract_filter):
    """
    crop2 - Remove points outside a given bounding box
        Arguments:
            minx: minimum X
            maxx: maximum X
            miny: minimum Y
            maxy: maximum Y
            minz: minimum Z
            maxz: maximum Z
    """
    filtername = "crop2"
    
    def __init__(self, minx : float, maxx : float, miny : float, maxy : float, minz : float, maxz : float):
        self.bounding_box = (minx, maxx, miny, maxy, minz, maxz)
        self.count = 0
        self.times = []
        self.original_pointcounts = []
        self.pointcounts = []
        self.last_rgb = None

    def extract_rgb_images(self, pc: cwipc_wrapper):
        """Extract and concatenate RGB images from the point cloud auxiliary data."""
        auxdata = pc.access_auxiliary_data()
        if not auxdata:
            print("crop2_rgb filter: no auxiliary data")
            return None
        per_camera_images = auxdata.get_all_images("rgb.")
        all_images = list(per_camera_images.values())
        if len(all_images) == 0:
            print("crop2_rgb: here 0 =.=")
            return None
        full_image = cv2.vconcat(all_images)

        # its a BGR image, convert to RGB
        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
        
        return full_image

    def filter(self, pc : cwipc_wrapper) -> cwipc_wrapper:
        self.count += 1
        t1_d = time.time()
        self.original_pointcounts.append(pc.count())

        x, y, w, h = 360, 310, 120, 150

        rgb_image = None
        rgb_image = self.extract_rgb_images(pc)
        # draw bounding box on rgb image
        # draw a dot on x, y
        if rgb_image is not None:
            cv2.rectangle(rgb_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(rgb_image, (x, y), 5, (0, 0, 255), -1)
        self.last_rgb = rgb_image

        points = pc.get_points()
        timestamp = pc.timestamp()
        cellsize = pc.cellsize()

        if len(points) == 0:
            t2_d = time.time()
            self.times.append(t2_d-t1_d)
            self.pointcounts.append(pc.count())
            return pc
        
        # convert for 4 corners
        x_corners = [x, x+w, x, x+w]
        y_corners = [y, y, y+h, y+h]
        # convert to phys coord
        front_depth = 1
        back_depth = 3
        bounding_corners = []
        for i in range(4):
            a, b, c = self.convert_depth_to_phys_coord_using_realsense(x_corners[i], y_corners[i], back_depth)
            bounding_corners.append((a, b, c))
            a, b, c = self.convert_depth_to_phys_coord_using_realsense(x_corners[i], y_corners[i], front_depth)
            bounding_corners.append((a, b, c))

        for i in range(len(bounding_corners)):
            points[i].x = bounding_corners[i][0]
            points[i].y = bounding_corners[i][1]
            points[i].z = bounding_corners[i][2]
            points[i].r = 255
            points[i].g = 0
            points[i].b = 0

        maxX = max([corner[0] for corner in bounding_corners])
        minX = min([corner[0] for corner in bounding_corners])
        maxY = max([corner[1] for corner in bounding_corners])
        minY = min([corner[1] for corner in bounding_corners])
        maxZ = max([corner[2] for corner in bounding_corners])
        minZ = min([corner[2] for corner in bounding_corners])

        self.bounding_box = (minX, maxX, minY, maxY, minZ, maxZ)
        # self.bounding_box = (0, 5, -5, 5, -5, 5)
        # print(f"crop2: bounding box: {(minX, maxX, minY, maxY, minZ, maxZ)}")
    
        # leave this point 0.0?
        # newpc = cwipc_from_points(points, timestamp)
        # newpc._set_cellsize(cellsize)
        # pc.free()
        # cropping cause the rgb window disappear
        cropped_pc = cwipc_crop(pc, self.bounding_box)
        pc.free()
        pc = cropped_pc

        t2_d = time.time()
        self.times.append(t2_d-t1_d)
        # self.pointcounts.append(newpc.count())
        # return newpc
    
        self.pointcounts.append(pc.count())
        return pc

    def convert_depth_to_phys_coord_using_realsense(self, x, y, depth, cameraInfo=None):
        if cameraInfo is None:
            _intrinsics = pyrealsense2.intrinsics()
            _intrinsics.width = 848
            _intrinsics.height = 480
            # _intrinsics.fx, _intrinsics.fy = 385.7314758300781, 385.2828369140625
            # _intrinsics.ppx, _intrinsics.ppy = 322.3908996582031, 239.84091186523438
            _intrinsics.fx = 431.63555908203125
            _intrinsics.fy = 431.63555908203125
            _intrinsics.ppx = 430.78387451171875
            _intrinsics.ppy = 237.67259216308594
            _intrinsics.model = pyrealsense2.distortion.brown_conrady
            _intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
            result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
            # return result[0], result[1], result[2] # cwipc use x, y, z
            # NEED transform to cwipc coord
            trafo = [
                [
                0.9982719346131126,
                -0.014794775046713896,
                -0.056870547694828744,
                0.03894209181601249
                ],
                [
                -0.018391640573880295,
                -0.9978280617400719,
                -0.06325272137269533,
                1.0453077987845374
                ],
                [
                -0.055811218592626055,
                0.06418935920670779,
                -0.9963758497895452,
                1.8533837414992975
                ],
                [
                0.0,
                0.0,
                0.0,
                1.0
                ]
            ]
            result = [result[0], result[1], result[2], 1.0]
            result = [sum([trafo[i][j] * result[j] for j in range(4)]) for i in range(4)]
            return result[0], result[1], result[2]
            return result[2], -result[0], -result[1]

        _intrinsics = pyrealsense2.intrinsics()
        _intrinsics.width = cameraInfo.width
        _intrinsics.height = cameraInfo.height
        _intrinsics.ppx = cameraInfo.K[2]
        _intrinsics.ppy = cameraInfo.K[5]
        _intrinsics.fx = cameraInfo.K[0]
        _intrinsics.fy = cameraInfo.K[4]
        #_intrinsics.model = cameraInfo.distortion_model
        _intrinsics.model  = pyrealsense2.distortion.none
        _intrinsics.coeffs = [i for i in cameraInfo.D]
        result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
        return result[2], -result[0], -result[1]

    def statistics(self) -> None:
        if self.times:
            self.print1stat('duration', self.times)
        if self.original_pointcounts:
            self.print1stat('original_pointcount', self.original_pointcounts, True)
        if self.pointcounts:
            self.print1stat('pointcount', self.pointcounts, True)
        
        if self.last_rgb is not None:
            self.save_last_rgb()

    def print1stat(self, name : str, values : Union[List[int], List[float]], isInt : bool=False) -> None:
        count = len(values)
        if count == 0:
            print(f'{self.filtername}: {name}: count=0')
            return
        minValue = min(values)
        maxValue = max(values)
        avgValue = sum(values) / count
        if isInt:
            fmtstring = '{}: {}: count={}, average={:.3f}, min={:d}, max={:d}'
        else:
            fmtstring = '{}: {}: count={}, average={:.3f}, min={:.3f}, max={:.3f}'
        print(fmtstring.format(self.filtername, name, count, avgValue, minValue, maxValue))
    
    def save_last_rgb(self):
        # Scale to something reasonable
        full_image = self.last_rgb

        # acutally no reshape
        # h, w, _ = full_image.shape
        # print(f"crop2_rgb: image shape: {w}x{h}")
        # hscale = 1024 / h
        # wscale = 1024 / w
        # scale = min(hscale, wscale)
        # if scale < 1:
        #     new_h = int(h*scale)
        #     new_w = int(w*scale)
        #     print(f"crop2_rgb: scaling to {new_w}x{new_h}")
        #     full_image = cv2.resize(full_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        cv2.imwrite("crop2_rgb.png", full_image)
        print(f"crop2_rgb: saved rgb image to crop2_rgb.png")

CustomFilter = Crop2Filter
