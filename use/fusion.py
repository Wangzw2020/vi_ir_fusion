# coding:utf-8
import os
import argparse
import time
import numpy as np
import time

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from FusionNet import FusionNet
from torch.autograd import Variable
from PIL import Image as P_Image
import cv2

import roslib
import rospy
from std_msgs.msg import Header
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image

import threading

#import sys
#sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')

scale_factor = 1.0

fusion_model_path = './model/fusionmodel_final.pth'
image_topic_1 = "/pub_t"
image_topic_2 = "/pub_rgb"
hightlight = False
vi_image_enhance = True

class FusionImage:
    def __init__(self):
    
        self.fusionmodel = eval('FusionNet')(output=1)
        self.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        if args.gpu >= 0:
            self.fusionmodel.to(self.device)
        self.fusionmodel.load_state_dict(torch.load(fusion_model_path))
        print('fusionmodel load done!')
        
        
        self._sub_irImage = rospy.Subscriber(image_topic_1, Image, self.callback_ir)
        self._sub_viImage = rospy.Subscriber(image_topic_2, Image, self.callback_vi)
        self._pub_fusion_image = rospy.Publisher("/pub_fusion", Image, queue_size=1)
        self.rate = 20
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate), self.callback_fusion)
        self.lock_ir = threading.Lock()
        self.lock_vi = threading.Lock()
        self.irImage = []
        self.viImage = []
        
    def callback_ir(self, image):
        self.lock_ir.acquire()
        img = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        
#        img = img[:,:,::-1]
        img = cv2.resize(img,None,fx=scale_factor, fy=scale_factor, interpolation = cv2.INTER_CUBIC)
        img = P_Image.fromarray(img)
        img = img.convert('L')
        img = np.array(img)
        img = np.asarray(P_Image.fromarray(img), dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        if len(self.irImage) > 0:
            self.irImage.pop(0)
            self.irImage.append(img)
        else:
            self.irImage.append(img)
        self.lock_ir.release()
        
    def callback_vi(self, image):
        self.lock_vi.acquire()
        img = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        
#        cv2.namedWindow("vi image",0)
#        cv2.imshow("vi image", img)
#        cv2.waitKey(1)
 
        if vi_image_enhance == True:
            img = image_enhance(img)
        
#        cv2.namedWindow("new image",0)
#        cv2.imshow("new image", img)
#        cv2.waitKey(1) 
#        img = img[:,:,::-1]
        img = cv2.resize(img,None,fx=scale_factor, fy=scale_factor, interpolation = cv2.INTER_CUBIC)
        
        if hightlight == True:
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            V = hsv[:,:,2]
            average = np.mean(V)
            max_V = 255 * np.sqrt((255 - average) / 255)
            mask = V > average
            V_a = (V - average) * (max_V - average) / (255 - average) + average
            V_a = V_a * mask
            mask = mask != True
            V_b = V * mask
            V = V_a + V_b
            V = V.astype(np.uint8)
            hsv_new = cv2.merge((hsv[:,:,0],hsv[:,:,1],V))
            img = cv2.cvtColor(hsv_new,cv2.COLOR_HSV2BGR)
        
        img = (
                np.asarray(P_Image.fromarray(img), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )
        if len(self.viImage) > 0:
            self.viImage.pop(0)
            self.viImage.append(img)
        else:
            self.viImage.append(img)
        self.lock_vi.release()
    
    def callback_fusion(self, event):
        
        self.lock_vi.acquire()
        if len(self.viImage) > 0:
            tmp_viImage = self.viImage[0]
            self.lock_vi.release()
        else:
            self.lock_vi.release()
            return
        
        self.lock_ir.acquire()
        if len(self.irImage) > 0:
            tmp_irImage = self.irImage[0]
            self.lock_ir.release()
        else:
            self.lock_ir.release()
            return
        
        print("start fusion!")
        
        irImage = torch.tensor(tmp_irImage)
        viImage = torch.tensor(tmp_viImage)
        
        viImage = viImage.unsqueeze(0)
        irImage = irImage.unsqueeze(0)

        with torch.no_grad():
            st = time.time()
            image_ir = Variable(irImage)
            image_vi = Variable(viImage)
            
            if args.gpu >= 0:
                image_vi = image_vi.to(self.device)
                image_ir = image_ir.to(self.device)
                
            image_vi_ycrcb = RGB2YCrCb(image_vi)
            
            logits = self.fusionmodel(image_vi_ycrcb, image_ir)
            fusion_ycrcb = torch.cat(
                (logits, image_vi_ycrcb[:, 1:2, :, :], image_vi_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)
            
            show_fused_image = fused_image[0,:,:,:]
            show_fused_image=show_fused_image[:,:,::-1]
            
            #print(show_fused_image.shape)
            cv2.namedWindow("fusion image",0)
            cv2.resizeWindow("fusion image", int(640 * 2 * scale_factor), int(480 * 2 * scale_factor))
            cv2.moveWindow("fusion image", 100 + int(640 * scale_factor), 100)
            cv2.imshow("fusion image", show_fused_image)
            cv2.waitKey(1)
            
            tmp_viImage = tmp_viImage.transpose(1,2,0)
            tmp_viImage=tmp_viImage[:,:,::-1]
            #print(tmp_viImage.shape)
            cv2.namedWindow("vi image",0)
            cv2.resizeWindow("vi image", int(640 * scale_factor), int(480 * scale_factor))
            cv2.moveWindow("vi image", 100, 100)
            cv2.imshow("vi image", tmp_viImage)
            cv2.waitKey(1) 
            
            tmp_irImage = tmp_irImage.transpose(1,2,0)
            tmp_irImage=tmp_irImage[:,:,::-1]
            #print(tmp_irImage.shape)
            cv2.namedWindow("ir image",0)
            cv2.resizeWindow("ir image", int(640 * scale_factor), int(480 * scale_factor))
            cv2.moveWindow("ir image", 100, 100 + int(480 * scale_factor))
            cv2.imshow("ir image", tmp_irImage)
            cv2.waitKey(1)
            
            image_temp = Image()
            header = Header(stamp=rospy.Time.now())
            header.frame_id = 'base_link'
            image_temp.encoding = 'rgb8'
            image_temp.height = int(480 * scale_factor)
            image_temp.width = int(640 * scale_factor)
            image_temp.step = int(640 * 3 * scale_factor)
            image_temp.data = np.array(fused_image).tostring()
            image_temp.header = header
            self._pub_fusion_image.publish(image_temp)
            
            ed = time.time()
            time_cost = ed - st
            print ("fusion time cost: ", time_cost)
            

def YCrCb2RGB(input_im):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (im_flat + bias).mm(mat).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def RGB2YCrCb(input_im):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    
    return out
    
def image_enhance(img):
#    r, g, b = cv2.split(img)
#    r1 = cv2.equalizeHist(r)
#    g1 = cv2.equalizeHist(g)
#    b1 = cv2.equalizeHist(b)
#    image_enhanced = cv2.merge([r1, g1, b1])
#    return image_enhanced

#    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#    image_enhanced = cv2.filter2D(img, cv2.CV_8UC3, kernel)
#    return image_enhanced

    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image_clahe = cv2.merge([b, g, r])
    return image_clahe
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SeAFusiuon with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='SeAFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    
    rospy.init_node('fusion_node')
    FusionImage()
    rospy.spin()
    
