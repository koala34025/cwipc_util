import time
from typing import Union, List
from .abstract import cwipc_abstract_filter
from ..util import cwipc_crop, cwipc_wrapper, cwipc_from_points
import cv2
import pyrealsense2
import numpy as np

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
        self.last_depth = None
        # 1000=1m, ?10cm
        # read in background depth image
        self.depth_background = cv2.imread("crop2_depth_background.png", cv2.IMREAD_UNCHANGED)
        print(f"crop2: background depth image shape: {self.depth_background.shape}")
        print(min(self.depth_background.flatten()), max(self.depth_background.flatten()))

    def extract_rgb_and_depth_images(self, pc: cwipc_wrapper):
        """Extract and concatenate RGB images from the point cloud auxiliary data."""
        auxdata = pc.access_auxiliary_data()
        if not auxdata:
            print("crop2_rgb filter: no auxiliary data")
            return None, None
        per_camera_images = auxdata.get_all_images("rgb.")
        rgb_image = list(per_camera_images.values())
        if len(rgb_image) == 0:
            print("crop2_rgb: here 0 =.=")
            return None, None
        rgb_image = cv2.vconcat(rgb_image)

        # its a BGR image, convert to RGB
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        depth_image = auxdata.get_all_images("depth.")
        depth_image = list(depth_image.values())
        if len(depth_image) == 0:
            print("crop2_rgb: here 1 =.=")
            return rgb_image, None
        depth_image = cv2.vconcat(depth_image)

        return rgb_image, depth_image

    def filter(self, pc : cwipc_wrapper) -> cwipc_wrapper:
        self.count += 1
        t1_d = time.time()
        self.original_pointcounts.append(pc.count())

        x, y, w, h = 360, 310, 120, 150

        rgb_image, depth_image = None, None
        rgb_image, depth_image = self.extract_rgb_and_depth_images(pc)
        # draw bounding box on rgb image
        # draw a dot on x, y
        if rgb_image is not None:
            cv2.rectangle(rgb_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(rgb_image, (x, y), 5, (0, 0, 255), -1)
        self.last_rgb = rgb_image
        self.last_depth = depth_image

        depth_diff = cv2.absdiff(self.depth_background, depth_image)
        _, binary_mask = cv2.threshold(depth_diff, 2000, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # find the largest contour
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            # print(area)
            if area > max_area:
                max_area = area
                max_contour = contour
        if max_contour is None:
            print("crop2: no contour found")
        print(max_area)

        if(max_area < 10000):
            # do not crop
            # print("crop2: no need to crop")    
            t2_d = time.time()
            self.times.append(t2_d-t1_d)
            self.pointcounts.append(pc.count())
            return pc

        # find the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(max_contour)

        points = pc.get_points()
        timestamp = pc.timestamp()
        cellsize = pc.cellsize()

        if len(points) < 10:
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
        # t2_d = time.time()
        # self.times.append(t2_d-t1_d)
        # self.pointcounts.append(newpc.count())
        # return newpc
    
        # cropping cause the rgb window disappear
        cropped_pc = cwipc_crop(pc, self.bounding_box)
        pc.free()
        pc = cropped_pc
        t2_d = time.time()
        self.times.append(t2_d-t1_d)
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
        if self.last_depth is not None:
            self.save_last_depth()

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
        rgb_image = self.last_rgb

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
        
        cv2.imwrite("crop2_rgb.png", rgb_image)
        print(f"crop2_rgb: saved rgb image to crop2_rgb.png")

    def save_last_depth(self):
        depth_image = self.last_depth
        # print shape
        print(f"crop2_depth: image shape: {depth_image.shape}")
        print(min(depth_image.flatten()), max(depth_image.flatten()))
        cv2.imwrite("crop2_depth.png", depth_image)
        print(f"crop2_depth: saved depth image to crop2_depth.png")

CustomFilter = Crop2Filter
