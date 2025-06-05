import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import speech_recognition as sr
import pyttsx3
from ultralytics import YOLO
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

# === Robot Arm Constants ===
a2 = 206.1553  # mm
a3 = 200.1662  # mm
d5 = 131       # mm

# === Trig helpers ===
cosd = lambda x: np.cos(np.deg2rad(x))
sind = lambda x: np.sin(np.deg2rad(x))
atan2d = lambda y, x: np.rad2deg(np.arctan2(y, x))

# === Inverse Kinematics ===
def compute_joint_angles(px, py, pz, phi_deg):
    r11 = cosd(phi_deg)
    r12 = sind(phi_deg)
    s234 = 0
    c234 = 1
    th234 = atan2d(s234, c234)
    th1 = atan2d(py, px)
    th5 = -atan2d(r12, r11) + th1

    try:
        Q = px / cosd(th1) if abs(cosd(th1)) > 0.10 else py / sind(th1)
    except ZeroDivisionError:
        return [np.nan] * 5

    A = Q - d5 * s234
    B = pz + d5 * c234
    c = (A**2 + B**2 + a2**2 - a3**2) / (2 * a2)
    a = A
    b = B

    discriminant = a**2 + b**2 - c**2
    if discriminant < -1e-6:
        return [np.nan] * 5

    discriminant = max(discriminant, 0)
    m = np.array([1, -1])
    TH2both = atan2d(b, a) + m * atan2d(np.sqrt(discriminant), c)
    th2 = np.max(TH2both)
    th23 = atan2d(B - a2 * sind(th2), A - a2 * cosd(th2))
    th3 = th23 - th2
    th4 = th234 - th23

    return [th1, th2, th3, th4, th5]

# === Adjust Angles for RX200 ===
def adjust_for_robot(th):
    if any(np.isnan(th)):
        return [np.nan] * 5
    th1, th2, th3, th4, th5 = th
    return np.array([
        th1,
        -(th2 - 76),
        -(th3 + 76),
        90 - th4,
        th5
    ]) * np.pi / 180

class ToolHandoverNode(Node):
    def __init__(self):
        super().__init__('tool_handover_node')
        self.bridge = CvBridge()
        self.bot = InterbotixManipulatorXS(robot_model="rx200")
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 150)
        self.subscription = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.depth_subscription = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.reset_state()

    def speak(self, text):
        print(f"🗣️ Robot says: {text}")
        self.tts.say(text)
        self.tts.runAndWait()
        time.sleep(0.1)


    def reset_state(self):
        self.image_msg = None
        self.depth_msg = None
        self.real_coords = {}
        self.safe_list = []
        self.hand_coords = None
        self.lookup = None

    def image_callback(self, msg):
        if self.image_msg is None:
            self.image_msg = msg

    def depth_callback(self, msg):
        if self.depth_msg is None:
            self.depth_msg = msg

    def capture_images(self):
        self.speak("Waiting for image from camera.")
        timeout = time.time() + 8
        while (self.image_msg is None or self.depth_msg is None) and time.time() < timeout:
            rclpy.spin_once(self, timeout_sec=0.5)
        return self.image_msg is not None and self.depth_msg is not None

    def run_detection(self):
        model = YOLO("lemon.pt")
        img = self.bridge.imgmsg_to_cv2(self.image_msg, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding='32FC1')
        results = model(img, save=False, show=False, conf=0.4)
        all_class_names = results[0].names
        contours_list = results[0].masks.xy
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        zc = 57.75 * 25.4
        tw = 45.1875* 25.4#47.5 or47.4
        img_h, img_w = img.shape[:2]
        pxcenter = 621
        pycenter = 357.3
        sft = tw / 720
        sf = sft / zc

        def pixels2xy(pxin, pyin, zin):
            x_mm = (pyin - pycenter) * sf * zin
            y_mm = (pxin - pxcenter) * sf * zin
            return x_mm, y_mm
        self.screw=False
        self.lemon=False
        for contour, cls_id in zip(contours_list, class_ids):
            class_name = all_class_names[cls_id]
            cnt = np.round(contour).astype(np.int32).reshape((-1, 1, 2))
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                raw_depth = float(depth_image[cy, cx]) if 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1] else zc
                if raw_depth == 0 or np.isnan(raw_depth):
                    raw_depth = zc
                x_mm, y_mm = pixels2xy(cx, cy, raw_depth)
                z_mm = -60 + (zc - raw_depth) + (20 if class_name == "hand" else 0)

                self.real_coords[class_name] = (x_mm, y_mm, z_mm)
                if class_name == "hand":
                    self.hand_coords = (x_mm, y_mm, z_mm)
                if "Safe" in class_name:
                    self.safe_list.append((x_mm, y_mm, z_mm))
                if class_name=="EyeScrew":
                    print(" Found the eye screw")
                    self.screw=True
                if class_name=="Lemon":
                    print(" Found a lemon!!!")
                    self.lemon=True
                

    def check_hand_reachability(self):
    # First, handle case where hand was not detected at all
     retries = 2
     while self.hand_coords is None and retries > 0:
        self.speak("Hand not detected. Please place your hand within the camera's view.")
        time.sleep(5)
        self.reset_state()
        if not self.capture_images():
            return False
        self.run_detection()
        retries -= 1

     if self.hand_coords is None:
        self.speak("No hand detected after multiple attempts. Exiting.")
        return False

    # Then check if the detected hand is within reach
     while True:
        hx, hy, hz = self.hand_coords
        phi_to_hand = np.degrees(np.arctan2(hy, hx))
        
        joint_to_hand = adjust_for_robot(compute_joint_angles(hx, hy, hz, phi_to_hand))
        if not any(np.isnan(joint_to_hand)):
            self.speak("Hand is within reach.")
            time.sleep(5)
            return True
        self.speak("Hand is out of reach. Please reposition your hand.")
        time.sleep(5)
        self.reset_state()
        if not self.capture_images():
            return False
        self.run_detection()

    
    def listen_for_command(self):
      
        
        if (self.screw):
            
            self.speak("Eye screw spotted. Handing screwdriver ")
            time.sleep(5)
            self.lookup="ScrewDriver"

        elif(self.lemon):
             
            self.speak("lemon spotted! handing Knife ")
            time.sleep(5)
            self.lookup="Knife"
        
        else:
          recognizer = sr.Recognizer()
          self.speak("Listening for knife or screwdriver.")
    
          with sr.Microphone() as source:
             recognizer.adjust_for_ambient_noise(source, duration=4)
             while True:
                try:
                    audio = recognizer.listen(source, timeout=2)
                    text = recognizer.recognize_google(audio).lower()
                    print(f"You said: {text}")
                    if "knife" in text:
                        self.lookup = "Knife"
                        break
                    elif "screwdriver" in text:
                        self.lookup = "ScrewDriver"
                        break
                    else:
                        print("No keyword detected. Please say knife or screwdriver.")
                        
                except:
                    print("I couldn't understand. Please try again.")

    def process_command(self):
        pick_key = f"{self.lookup}Pick"
        if pick_key in self.real_coords and self.safe_list:
            p_pick = np.array(self.real_coords[pick_key])
            safe_array = np.array(self.safe_list)
            distances = np.linalg.norm(safe_array[:, :2] - p_pick[:2], axis=1)
            p_safe = safe_array[np.argmin(distances)]
            vec = p_safe[:2] - p_pick[:2]
            phi_deg = np.degrees(np.arctan2(vec[1], vec[0]))
            self.move_to_position(p_pick[0], p_pick[1], p_pick[2], phi_deg)
        else:
            self.speak("Required tool not found in image.")
            time.sleep(5)

    def move_to_position(self, x, y, z, phi):
        via_z = 0
        self.bot.gripper.set_pressure(1.0)
        via_command = adjust_for_robot(compute_joint_angles(x, y, via_z, phi))
        final_command = adjust_for_robot(compute_joint_angles(x, y, z, phi))
        handover_command = adjust_for_robot(compute_joint_angles(250, 0, 150, 0))

        self.bot.arm.go_to_home_pose()
        self.bot.gripper.release()
        time.sleep(1)
        self.bot.arm.set_joint_positions(via_command)
        time.sleep(1)
        self.bot.arm.set_joint_positions(final_command)
        time.sleep(1)
        self.bot.gripper.grasp()
        time.sleep(1)
        self.bot.arm.set_joint_positions(via_command)
        time.sleep(1)
        self.bot.arm.set_joint_positions(handover_command)
        time.sleep(2)

        if self.hand_coords:
            hx, hy, hz = self.hand_coords
            phi_to_hand = np.degrees(np.arctan2(hy, hx))
            joint_up_hand=adjust_for_robot(compute_joint_angles(hx, hy, hz+70, phi_to_hand))
            joint_to_hand = adjust_for_robot(compute_joint_angles(hx, hy, hz, phi_to_hand))
            if not any(np.isnan(joint_to_hand)):
                print("Moving to hand for transfer.")
                self.bot.arm.set_joint_positions(joint_up_hand)
                time.sleep(2)
                self.bot.arm.set_joint_positions(joint_to_hand)
                time.sleep(2)
                

        self.bot.gripper.release()
        time.sleep(1)
        self.bot.arm.set_joint_positions(joint_up_hand)
        time.sleep(2)
        self.bot.arm.go_to_home_pose()
        time.sleep(1)
        self.bot.arm.go_to_sleep_pose()
        self.speak("Tool handed over successfully.")
        time.sleep(6)

    def ask_continue(self):
        recognizer = sr.Recognizer()
        self.speak("Would you like something else? Say yes or stop.")
        time.sleep(7)
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=5)
            while True:
                try:
                    audio = recognizer.listen(source, timeout=5)
                    text = recognizer.recognize_google(audio).lower()
                    print(f"You said: {text}")
                    if "yes" in text:
                        self.speak("Okay. Let's go again.")
                        time.sleep(3)
                        return True
                    elif "stop" in text:
                        self.speak("Goodbye.")
                        time.sleep(3)
                        return False
                    else:
                        self.speak("Please say yes to continue or stop to quit.")
                        time.sleep(10)
                except:
                    self.speak("I couldn't understand. Please say yes or stop.")
                    time.sleep(5)

def main(args=None):
    rclpy.init(args=args)
    node = ToolHandoverNode()
    try:
        while rclpy.ok():
            node.reset_state()
            if not node.capture_images():
                break
            node.run_detection()
            if not node.check_hand_reachability():
                break
            node.listen_for_command()
            node.process_command()
            if not node.ask_continue():
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()