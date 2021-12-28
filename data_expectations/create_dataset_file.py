from PIL import Image

def create(filename,dataset):

    with open(filename,"w") as f:
        
        f.write("Image,Format,Mode,Height,Width,IsCorrupted\n")
        
        for image_name in dataset:
            
            try:
                
                image = Image.open(image_name)
                format = image.format 
                mode = image.mode 
                height, width = str(image.size[0]), str(image.size[1])

                data = ','.join([image_name,format,mode,height,width,"0"])
                f.write(data + "\n")     
                
            except Exception as e:

                data = ','.join([image_name,"JPEG","RGB","40","40","1"])
                f.write(data + "\n")  
                
               
"""                
images_face_detection_train = glob.glob("dataset/yolo/train/images/*")
images_face_detection_valid = glob.glob("dataset/yolo/valid/images/*")
images_face_detection_test = glob.glob("dataset/yolo/test/images/*")     
                
create("images_face_detection_train.csv",images_face_detection_train)                
create("images_face_detection_valid.csv",images_face_detection_valid)                
create("images_face_detection_test.csv",images_face_detection_test)                
"""
