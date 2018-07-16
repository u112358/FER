import numpy as np
import os
import matplotlib.image as mpimg


class image_processor():
    def __init__(self, database='../data'):
        # self.DATA_DIR = "C:\\Users\\IB\\Desktop\\Dissertation\\Datasets\\IBUG"
        # self.data_dir = 'path\to\your\dataset'
        self.data_dir = database
        self.img_list = [v for v in os.listdir(self.data_dir) if v.__contains__('png')]
        self.total_image_number = len(self.img_list)
        self.current_index = 0

    """
    Open and read the .pts document to create a list of landmark coordinates
    """

    def read_lmarks(slef, file_name=None):
        lmarks = []
        with open(file_name) as file:
            i = 0
            for line in file:
                if "version" in line or "points" in line or "{" in line or "}" in line:
                    continue
                else:
                    x_t, y_t = line.strip().split(" ")
                    lmarks.append([float(x_t), float(y_t)])
                i += 1
        return lmarks

    """
    Extract patch images around the loc
    """

    def get_patches(self, image, loc):
        patches = []
        for i in range(68):
            # The min & max operations are employed here to make sure the index won't exceed the range.
            x = int(round(float(loc[i][0])))
            x = min(x,1024-32)
            x = max(x,32)
            y = int(round(float(loc[i][1])))
            y = min(y,1024-32)
            y = max(y,32)
            patches.append(image[x - 32:x + 32, y - 32:y + 32, :])
        patches = np.reshape(patches,[-1,64,64,3])
        return patches

    '''
    Get next training image and its corresponding ground truth
    '''

    def get_next_image_and_gt(self):  ###
        self.current_index = np.random.randint(1, self.total_image_number)
        self.img_path = '%s/indoor_%03d.png' % (self.data_dir, self.current_index)
        self.pts_path = '%s/indoor_%03d.pts' % (self.data_dir, self.current_index)
        image = mpimg.imread(self.img_path)
        gt = self.read_lmarks(self.pts_path)

        return image, gt

    '''
        Get faked training image and its corresponding ground truth to test our model, can be abandoned in final version
    '''

    def get_fake_next_image_and_gt(self):  ###
        image = np.random.randint(0, 255, (1024, 1024, 3))
        gt = np.random.randint(256, 512, (68, 2))
        return image, gt

    '''
    Complete yourself to create an initial locations
    '''

    def get_init_loc(self):
        return np.random.randint(256, 512, (68, 2))

# uncomment to test the image_processor
# if __name__ == '__main__':
#     IP = image_processor()
#     image, gt = IP.get_next_image_and_gt()
#     patches = IP.get_patches(image, gt)
#     print IP.get_init_loc()
#     print('done')
