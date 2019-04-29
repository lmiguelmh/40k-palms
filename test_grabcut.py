import cv2
import numpy as np
import sys
import os

COLOR_BG = (255,0,0)
COLOR_FG = (0,255,0)

def mask2color(mask):
    r,c = mask.shape[:2]
    color = np.zeros((r,c,3),np.uint8)
    color[np.where((mask==0)|(mask==2))] = COLOR_BG
    color[np.where((mask==1)|(mask==3))] = COLOR_FG
    return color

def color2mask(color):
    r,c = color.shape[:2]
    mask = np.zeros((r,c),np.uint8)
    mask[np.where((color==COLOR_BG).all(axis=2))] = 0
    mask[np.where((color==COLOR_FG).all(axis=2))] = 1
    return mask

def on_mouse(event,x,y,flags,param):
    param.mouse_cb(event,x,y,flags)

def nothing(x):
    pass

class InteractiveImageSegmentation:
    def __init__(self):
        self.winname = "InteractiveImageSegmentation"
        self.img = np.zeros((0))
        self.mask = np.zeros((0))
        self.left_mouse_down = False
        self.right_mouse_down = False
        self.radius = 3
        self.max_radius = 40
        self.use_prev_mask = False
        self.cur_mouse = (-1,-1)
        self.draw_color = 0
        cv2.namedWindow(self.winname)
        cv2.setMouseCallback(self.winname, on_mouse, self)
        cv2.createTrackbar('brush size',self.winname,self.radius,self.max_radius,nothing)

    def mouse_cb(self,event,x,y,flags):
        self.cur_mouse = (x,y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_mouse_down = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.left_mouse_down = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.right_mouse_down = True
        elif event == cv2.EVENT_RBUTTONUP:
            self.right_mouse_down = False
        if (self.left_mouse_down or self.right_mouse_down) and self.mask.size>0 and self.img.size>0:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                cv2.circle(self.img, (x,y), self.radius, (COLOR_BG if self.left_mouse_down else tuple([k/3 for k in COLOR_BG])), -1)
                cv2.circle(self.mask, (x,y), self.radius, (cv2.GC_BGD if self.left_mouse_down else cv2.GC_PR_BGD), -1)
            elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                cv2.circle(self.img, (x,y), self.radius, (COLOR_FG if self.left_mouse_down else tuple([k/3 for k in COLOR_FG])), -1)
                cv2.circle(self.mask, (x,y), self.radius, (cv2.GC_FGD if self.left_mouse_down else cv2.GC_PR_FGD), -1)
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags<0:
                diff_k = int(np.clip(self.radius*0.4,1,5))
                self.radius+=diff_k
            elif flags>0:
                diff_k = int(np.clip(self.radius*0.4,1,5))
                self.radius-=diff_k
            self.radius = np.clip(self.radius, 1, self.max_radius)
            cv2.setTrackbarPos('brush size', self.winname, self.radius)

    def __init_mask(self, mask):
        mask[:] = cv2.GC_PR_FGD
        mask[:10,:] = cv2.GC_PR_BGD

    def process(self, img):
        self.img = np.copy(img)
        if self.use_prev_mask==False or self.mask.shape[:2]!=self.img.shape[:2]:
            self.mask = np.zeros(img.shape[:2],'uint8')
            self.__init_mask(self.mask)
        self.bgdModel = np.zeros((1,65),np.float64)
        self.fgdModel = np.zeros((1,65),np.float64)
        cv2.grabCut(img, self.mask, None, self.bgdModel, self.fgdModel, 1, cv2.GC_INIT_WITH_MASK)

        while True:
            self.radius = cv2.getTrackbarPos('brush size',self.winname)
            color = mask2color(self.mask)
            alpha = 0.5 if self.draw_color==0 else (1 if self.draw_color==1 else 0)
            show_img = (self.img*alpha + color*(1-alpha)).astype('uint8')
            cv2.circle(show_img, self.cur_mouse, self.radius, (200,200,200), (2 if self.left_mouse_down else 1))
            cv2.imshow(self.winname,show_img)
            cv2.imshow('color',color)
            key = cv2.waitKey(100)
            if key == ord('c'):
                self.img = np.copy(img)
                self.__init_mask(self.mask)
            elif key == ord('q') or key == 27 or key==ord('s') or key==ord('p') or key==ord('n') or key == 10:
                break
            elif key == ord('w'):
                self.draw_color = (self.draw_color+1)%3
            elif key == ord('a') or key == 32:
                cv2.putText(show_img, 'segmenting...', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2)
                cv2.imshow(self.winname,show_img)
                cv2.waitKey(1)
                cv2.grabCut(img, self.mask, None, self.bgdModel, self.fgdModel, 1, cv2.GC_INIT_WITH_MASK)
                self.img = np.copy(img)
        return key

if __name__ == '__main__':
    if(len(sys.argv)!=2):
        print('Usage: interactive_image_segmentation.py [img]')
        exit()

    img = sys.argv[1]
    # save_dir = sys.argv[2]
    maskimg = 'mask-' + img

    if not os.path.exists(img):
        print('%s not exists.'%img)
        exit()

    iis = InteractiveImageSegmentation()
    iis.use_prev_mask = True
    key = iis.process(cv2.imread(img))
    if key == ord('s') or key == 10:
        cv2.imwrite(maskimg,mask2color(iis.mask))
        print('save label %s.'%maskimg)
    elif key == ord('p'):
        print('p')
    elif key == ord('n') or key == 32:
        print('n')
    elif key == ord('q') or key == 27:
        print('q')

    iis.mask[np.where(iis.mask==cv2.GC_BGD)]=cv2.GC_PR_BGD
    iis.mask[np.where(iis.mask==cv2.GC_FGD)]=cv2.GC_PR_FGD

