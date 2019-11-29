

def scale_filter(scale1 , scale2,scale_threshold,key):
    #if(scale1 == scale2):
        #scale_class1 = key + "_1"
        #scale_class2 = key + "_1"

    if( scale2 - 0.35  <= scale1 <= scale2 + 0.35 ):
        if(scale1 <= scale_threshold  ):
            scale_class1 = key + "_1"
        else:
            scale_class1 = key + "_2"
        
        
        if(scale2 <= scale_threshold):
            scale_class2 = key + "_1"
        else:
            scale_class2 = key + "_2"
    else:
        if(scale1 <= scale_threshold  ):
            scale_class1 = key + "_1"
        else:
            scale_class1 = key + "_2"

        if(scale2 <= scale_threshold):
            scale_class2 = key + "_1"
        else:
            scale_class2 = key + "_2"


    return(scale_class1, scale_class2)
    
def class_filter(scale1 , scale2,scale_threshold,key):
    #if(scale1 == scale2):
        #scale_class1 = key + "_1"
        #scale_class2 = key + "_1"

    if( scale2 - 0.35  <= scale1 <= scale2 + 0.35 ):
        if(scale1 <= scale_threshold  ):
            scale_class1 = key + "_1"
        else:
            scale_class1 = key + "_2"
        
        
        if(scale2 <= scale_threshold):
            scale_class2 = key + "_1"
        else:
            scale_class2 = key + "_2"
    else:
        if(scale1 <= scale_threshold  ):
            scale_class1 = key + "_1"
        else:
            scale_class1 = key + "_2"

        if(scale2 <= scale_threshold):
            scale_class2 = key + "_1"
        else:
            scale_class2 = key + "_2"


    return(scale_class1, scale_class2)
    
#def offset_filter(offset1 , offset2):    
def offset_filter(offset1 , offset2,offset_threshold,key):

    if( offset2 - 0.35  <= offset1 <= offset2 + 0.35 ):
        if(offset1 <= offset_threshold  ):
            offset_class1 = key + "_1"
        else:
            offset_class1 = key + "_2"
        
        
        if(offset2 <= offset_threshold):
            offset_class2 = key + "_1"
        else:
            offset_class2 = key + "_2"
    else:
        if(offset1 <= offset_threshold  ):
            offset_class1 = key + "_1"
        else:
            offset_class1 = key + "_2"

        if(offset2 <= offset_threshold):
            offset_class2 = key + "_1"
        else:
            offset_class2 = key + "_2"


    return(offset_class1, offset_class2)
    

    