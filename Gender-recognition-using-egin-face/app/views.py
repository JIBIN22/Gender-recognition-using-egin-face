import os
import cv2
from app.face_recognition import faceRecognitionPipeline
from flask import render_template, request
import matplotlib.image as matimg


UPLOAD_FOLDER = 'static/upload'

def index():
    return render_template('index.html')


def app():
    return render_template('app.html')


def genderapp():
    if request.method == 'POST':
            print("file:")
            try:
                if '.' in request.files['image_name'].filename:
                    extension = request.files['image_name'].filename.rsplit('.', 1)[1].lower()
                    if extension in ['jpg', 'jpeg', 'png', 'gif']:
                        f = request.files['image_name']
                        filename = f.filename
                        # save our image in upload folder
                        path = os.path.join(UPLOAD_FOLDER,filename)
                        f.save(path) # save image into upload folder
                        # get predictions
                        pred_image, predictions = faceRecognitionPipeline(path)
                        pred_filename = 'prediction_image.jpg'
                        cv2.imwrite(f'./static/predict/{pred_filename}',pred_image)
                        
                        # generate report
                        report = []
                        for i , obj in enumerate(predictions):
                            gray_image = obj['roi'] # grayscale image (array)
                            eigen_image = obj['eig_img'].reshape(100,100) # eigen image (array)
                            gender_name = obj['prediction_name'] # name 
                            score = round(obj['score']*100,2) # probability score
                            
                            # save grayscale and eigne in predict folder
                            gray_image_name = f'roi_{i}.jpg'
                            eig_image_name = f'eigen_{i}.jpg'
                            matimg.imsave(f'./static/predict/{gray_image_name}',gray_image,cmap='gray')
                            matimg.imsave(f'./static/predict/{eig_image_name}',eigen_image,cmap='gray')
                            
                            # save report 
                            report.append([gray_image_name,
                                        eig_image_name,
                                        gender_name,
                                        score])
                            
                        
                        return render_template('gender.html',fileupload=True,report=report) # POST REQUEST
                    elif extension in ['mp4', 'avi', 'mov', 'mkv']:
                        f=request.files['image_name']
                        filename = f.filename
                        path = os.path.join(UPLOAD_FOLDER,filename)
                        print("elseif",str(f))
                        cap = cv2.VideoCapture(path) # 0 -> access webcamera, 1-> external camera, path
                        
                        while True:
                            ret, frame = cap.read()
                            
                            if ret == False:
                                break
                            
                            pred_img, pred_dict = faceRecognitionPipeline(frame,path=False)
                            
                            cv2.imshow('prediction',pred_img)
                            if cv2.waitKey(1) == ord('q'):
                                break
                            else:
                                cv2.setWindowProperty('prediction', cv2.WND_PROP_TOPMOST, 1)
                        

                        cap.release()
                        cv2.destroyAllWindows() 
                        return render_template('gender.html',fileupload=False) # GET REQUEST"""
            
    
            except KeyError:    
                cap = cv2.VideoCapture(0) # 0 -> access webcamera, 1-> external camera, path
                while True:
                    ret, frame = cap.read()
                    
                    if ret == False:
                        break
                    
                    pred_img, pred_dict = faceRecognitionPipeline(frame,path=False)
                    
                    cv2.imshow('prediction',pred_img)
                    if cv2.waitKey(1) == ord('q'):
                        break
                    else:
                        cv2.setWindowProperty('prediction', cv2.WND_PROP_TOPMOST, 1)
                

                cap.release()
                cv2.destroyAllWindows() 
                return render_template('gender.html',fileupload=False) # GET REQUEST"""
            
    
            
    return render_template('gender.html',fileupload=False) # GET REQUEST

