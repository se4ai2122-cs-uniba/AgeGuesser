import great_expectations as ge
import pandas as pd

def test_ge(dataset):
     
    print("\n" + dataset + "\n")
    
    df = ge.dataset.PandasDataset(pd.read_csv(dataset))
    
    results = list()
    
    fields = ["Image","Format","Mode","Height","Width","IsCorrupted"]
        
    results.append(df.expect_column_values_to_be_unique("Image",result_format={'result_format':'COMPLETE'}))   

    results.append(df.expect_column_values_to_match_regex_list("Format",["^JPEG$"],result_format={'result_format':'COMPLETE'}))
    
    results.append(df.expect_column_values_to_match_regex_list("Mode",["^RGB$"],result_format={'result_format':'COMPLETE'}))  
    
    results.append(df.expect_column_values_to_be_between("Height",30,result_format={'result_format':'COMPLETE'}))  

    results.append(df.expect_column_values_to_be_between("Width",30,result_format={'result_format':'COMPLETE'}))  
    
    results.append(df.expect_column_values_to_be_between("IsCorrupted",0,0,result_format={'result_format':'COMPLETE'}))  
    
    for field, result in zip(fields,results):
        
        print("\n\n" + field + "\n\n")
        
        print(result)

"""
test_ge("images_face_detection_train.csv")
test_ge("images_face_detection_valid.csv")
test_ge("images_face_detection_test.csv")
"""





