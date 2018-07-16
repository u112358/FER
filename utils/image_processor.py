class image_processor():
    def __init__(self):
        #self.DATA_DIR = "C:\\Users\\IB\\Desktop\\Dissertation\\Datasets\\IBUG"
        self.data_dir = 'path\to\your\dataset'

    """
    Open and read the .pts document to create a list of landmark coordinates
    """
    def read_lmarks(slef,file_name=None):
        lmarks = []
        with open(file_name) as file:
            i = 0
            for line in file:
                if "version" in line or "points" in line or "{" in line or "}" in line:
                    continue
                else:
                    x_t, y_t = line.strip().split(sep=" ")
                    lmarks.append([float(x_t), float(y_t)])
                i += 1
        return lmarks

    """
    Create a list of 68 patches with a size of 58x58
    """
    def get_patches(self, image,loc):
        patches=[]
        for i in range(68):
            patches.append(image[loc[i][0]-29:image[loc[i][0]+29],loc[i][1]-29:loc[i][1]+29])
        return patches

    def get_images(self, file_path, ): ###
        get image from file_path

