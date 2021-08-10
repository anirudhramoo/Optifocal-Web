import cv2, dlib, math
import numpy as np

from utils.EulerAng import faceIndex, make3d, make2d, get_euler_angle

class Counter():
    def __init__(self, frames, ratio = 0.6) -> None:
        self.thres = int(frames * ratio)
        self.frames = frames
        self.count0 = 0
        self.count1 = 0

    def update(self, cond_0, cond_1 = False):
        if cond_0:
            if self.count0 < self.frames:
                self.count0 += 1
            if self.count1 > 0:
                self.count1 -= 1
        elif cond_1:
            if self.count1 < self.frames:
                self.count1 += 1
            if self.count0 > 0:
                self.count0 -= 1
        else:
            self.decrement()
        
        if (self.count0 > self.thres):
            return 1
        else:
            return -(self.count1 > self.thres)

    def decrement(self):
        if self.count1 > 0:
            self.count1 -= 1
        if self.count0 > 0:
            self.count0 -= 1

    def display(self, labe = ""):
        print(labe, self.thres, self.frames, self.count0, self.count1)

    def reset(self) -> None:
        self.count0 = 0
        self.count1 = 0


class Calibrator:
    def __init__(self, limt, thres, name = "") -> None:
        self.limt = limt
        self.thres = thres
        self.consec = 0
        self.min = math.inf
        self.max = -math.inf
        self.sum = 0
        self.name = name

    def update(self, val):
        
        self.sum+=val
        if val > self.max:
            self.max = val
        
        elif val < self.min:
            self.min = val
        
        if self.max-self.min >= self.thres:
            print(self.name, "Threshold Exceeded")
            self.reset()
            
        self.consec += 1  
        if self.consec >= self.limt:
            return self.sum / self.consec

        return None

    def display(self, labl = ""):
        print(labl, self.consec, self.sum, self.min, self.max)

    def reset(self):
        self.consec = 0  
        self.sum=0 
        self.min = math.inf
        self.max = -math.inf



class attn_detector:
    ## SECTION 1: GENERAL FACE DETECTION
    
    def not_in(self, tLeft, bRight):
        return (tLeft[0]<self.xmin or tLeft[1]<self.ymin or bRight[0]>self.xmax or bRight[1]>self.ymax)


    ## SECTION 2: POSE DETECTION

    def updt_pose(self, shape):
        image_points = make2d(shape)

        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, image_points,self.camera_matrix, self.dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)
        
        # Comment out, only for debugging
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, self.camera_matrix, self.dist_coeffs)
        self.p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        self.p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        #result=str(p2[0])+" "+str(p2[1])
        #cv2.line(self.img, self.p1, self.p2, (255,0,0), 2)
        #print(self.p1, self.p2)
        
        return get_euler_angle(rotation_vector)

    def check_pose(self, pitch, yaw, roll):

        # Comment out, only for debugging
        #pose_str = "Pitch:{:.2f}, Yaw:{:.2f}, Roll:{:.2f}".format(pitch, yaw, roll)
        #self.put_text(pose_str, (25, 80), (0,255,0))
        #self.consec_hori.display("Horizontal:")
        #self.consec_vert.display("Vertical:")
            
        return self.consec_hori.update(yaw<-30, yaw > 30), self.consec_vert.update(pitch > 10, pitch < -10)


    ## SECTION 3: INDEPENDENT FUNCTIONS
    # TOBEDONE


    ## SECTION 4: UPDATING THE IMAGE AND ACTUALLY CHECKING POSE/GAZE/ECT

    def calibrate(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray) # , 1) # adding this second argument detects faces better, but is significantyl slower
        biggestface = faceIndex(faces)

        #calib_hori.display()
        #calib_vert.display()

        if biggestface < 0:
            #self.put_text("FACE NOT FOUND", (25, 40), (0,255,0))
            print("Face not Found")
            self.calib_hori.reset()
            self.calib_vert.reset()
            return -3, -3, 0, 0, 0, 0, 0
        
        else:
            face = faces[biggestface]
            
            if self.not_in((face.left(), face.top()), (face.right(), face.bottom())):
                print("Out of Frame")
                self.calib_hori.reset()
                self.calib_vert.reset()
                return -3, -3, 0, 0, 0, 0, 0

            else:
                shape = self.predictor(gray, face)
                ret, pitch, yaw, roll = self.updt_pose(shape)
                pitch = 180 - pitch if pitch > 0 else -180 - pitch

                # Comment out, only for debugging
                #pose_str = "Pitch:{}, Yaw:{:}, Roll:{:}".format(pitch, yaw, roll)
                #self.put_text(pose_str, (25, 80), (0,255,0))
                
                self.base_yaw = self.calib_hori.update(yaw)
                self.base_pitch = self.calib_vert.update(pitch)

                if not (self.base_yaw == None or self.base_pitch == None):
                    self.propogate = self.run

                return -2, -2, pitch, yaw, roll, self.p1, self.p2
    

    def run(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray )#, 1) # adding this second argument detects faces better, but is significantyl slower
        biggestface = faceIndex(faces)
        
        if biggestface < 0:
            #self.put_text("FACE NOT FOUND", (25, 40), (0,255,0))

            print("Face not found")
            return -3, -3, 0, 0, 0, 0, 0
        else:
            face = faces[biggestface]
            shape = self.predictor(gray, face)

            ret, pitch, yaw, roll = self.updt_pose(shape)

            pitch = 180 - pitch if pitch > 0 else -180 - pitch


            yaw -= self.base_yaw
            pitch -= self.base_pitch

            Horizontal, Vertical = self.check_pose(pitch,yaw,roll)
            
            # if not (Vertical or Horizontal):
            #     ear_left = self.eye_aspect_ratio(shape, self.left)
            #     ear_right = self.eye_aspect_ratio(shape, self.right)
            #     ear_avg = (ear_left + ear_right)/2
            #     Gaze = self.check_eyes(ear_avg, shape, gray)

            # Comment out for debugging purposes only
            # self.put_text("HORI: " + self.Hoript[Horizontal], (25, 190))
            # self.put_text("VERT: " + self.Vertpt[Vertical], (25, 230))

            return Horizontal, Vertical, pitch, yaw, roll, self.p1, self.p2


    ## SECTION 6: INIT

    def decodeimg(self, img_bin):
        return cv2.imdecode(np.frombuffer(img_bin, np.uint8), cv2.IMREAD_COLOR)

    def happen(self, imgstr):
        self.img = self.decodeimg(imgstr)
        return self.propogate()

    def reset(self):
        self.calib_hori.reset()
        self.calib_vert.reset()

        self.consec_gaze.reset()
        self.consec_hori.reset()
        self.consec_vert.reset()

        self.update = self.cam_init

    def cam_init(self, imgstr):
        self.img = self.decodeimg(imgstr)
        size = self.img.shape
        
        self.xmin = (size[1]//10)
        self.xmax = self.xmin * 9
        self.ymin = (size[0]//10)
        self.ymax = self.ymin * 9

        self.focal_length = size[1]
        self.center = (size[1]/2, size[0]/2)
        self.camera_matrix = np.array([[self.focal_length, 0, self.center[0]],
                                        [0, self.focal_length, self.center[1]],
                                        [0, 0, 1]], dtype = "double"
                                        )

        #PREREQS DONE
        self.propogate = self.calibrate
        self.update = self.happen

        return -3, -3, 0, 0, 0, 0, 0

    def __init__(self) -> None:
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("resc/shape_68.dat")

        self.model_points = make3d()
        self.dist_coeffs = np.zeros((4,1))
        self.left = [36, 37, 38, 39, 40, 41]
        self.right = [42, 43, 44, 45, 46, 47]
        self.gaze_ratio = 1

        self.calib_vert = Calibrator(35, 10, "Vertical")
        self.calib_hori = Calibrator(35, 25, "Horizontal")

        self.consec_gaze = Counter(70)
        self.consec_hori = Counter(25)
        self.consec_vert = Counter(25)

        # self.Vertpt = {0 : "CENTER", 1 : "UP", -1 : "DOWN"}                         # Looking up and down (pitch)
        # self.Hoript = {0 : "CENTER", 1 : "LEFT", -1 : "RIGHT"}                      # Looking left and right (yaw)

        self.base_yaw = 0
        self.base_pitch = 0

        self.update = self.cam_init
