import numpy as np
import random
from torch.utils.data import Dataset
from feeders import tools
from feeders import aug
import pickle


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', simpling='center_crop',
                 random_choose=False, random_shift=False, random_move=False, random_rot=False, 
                 window_size=-1, normalization=False, debug=False, use_mmap=True, use_aug=False,
                 bone=False, vel=False):
        """
        data_path:
        label_path:
        split: training set or test set
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move:
        random_rot: rotate skeleton around xyz axis
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
        use_mmap: If true, use mmap mode to load data, which can save the running memory
        bone: use bone modality or not
        vel: use motion modality or not
        only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split

        self.simpling = simpling

        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.use_aug = use_aug
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        if self.use_mmap:
            npz_data = np.load(self.data_path, mmap_mode='r')
        else:
            npz_data = np.load(self.data_path)

        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')

        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

        if self.debug:
            self.data = self.data[:100]
            self.label = self.label[:100]
            self.sample_name = self.sample_name[:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)        
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        # crop
        if self.simpling == "center_crop":
            data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        elif self.simpling == "global_crop":
            data_numpy = tools.crop_resize(data_numpy, valid_frame_num, self.window_size)
        else:
            raise ValueError(f"The value of simpling must be in center_crop or global_crop, but {self.simpling} now")
        if self.random_rot:
            # data_numpy = tools.random_rot(data_numpy)
            data_numpy = random_rot_v2(data_numpy)
        # if self.bone:
        #     ntu_pairs = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
        #         (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        #         (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
        #         (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12))
        #     bone_data_numpy = np.zeros_like(data_numpy)
        #     for v1, v2 in ntu_pairs:
        #         bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
        #     data_numpy = bone_data_numpy
        # if self.vel:
        #     data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
        #     data_numpy[:, -1] = 0

        if self.use_aug:
            data_numpy = self._aug(data_numpy)

        return data_numpy, label, index
    def _aug(self, data_numpy):
        if random.random() < 0.5:
            data_numpy = aug.Flip(data_numpy)
        if random.random() < 0.5:
            data_numpy = aug.Shear(data_numpy)
        return data_numpy


class Feeder_semi(Dataset):
    def __init__(self, data_path, label_path=None, label_percent=-1, p_interval=1, split='train', simpling='center_crop',
                 random_choose=False, random_shift=False, random_move=False, random_rot=False, 
                 window_size=-1, normalization=False, debug=False, use_mmap=True,
                 bone=False, vel=False):
        """
        data_path:
        label_path:
        split: training set or test set
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move:
        random_rot: rotate skeleton around xyz axis
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
        use_mmap: If true, use mmap mode to load data, which can save the running memory
        bone: use bone modality or not
        vel: use motion modality or not
        only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.label_percent = label_percent

        self.simpling = simpling

        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        if self.use_mmap:
            npz_data = np.load(self.data_path, mmap_mode='r')
        else:
            npz_data = np.load(self.data_path)

        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')

        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

        if self.label_percent > 0 and self.label_percent <= 1:
            n = len(self.label)
            # Record each class sample id
            class_blance = {}
            for i in range(n):
                if self.label[i] not in class_blance:
                    class_blance[self.label[i]] = [i]
                else:
                    class_blance[self.label[i]] += [i]

            final_choise = []
            for c in class_blance:
                c_num = len(class_blance[c])
                choise = random.sample(class_blance[c], round(self.label_percent * c_num))
                final_choise += choise
            final_choise.sort()

            self.data = self.data[final_choise]
            new_sample_name = []
            new_label = []
            for i in final_choise:
                new_sample_name.append(self.sample_name[i])
                new_label.append(self.label[i])

            self.sample_name = new_sample_name
            self.label = new_label

        if self.debug:
            self.data = self.data[:100]
            self.label = self.label[:100]
            self.sample_name = self.sample_name[:100]


    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)        
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        # crop
        if self.simpling == "center_crop":
            data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        elif self.simpling == "global_crop":
            data_numpy = tools.crop_resize(data_numpy, valid_frame_num, self.window_size)
        else:
            raise ValueError(f"The value of simpling must be in center_crop or global_crop, but {self.simpling} now")
        if self.random_rot:
            # data_numpy = tools.random_rot(data_numpy)
            data_numpy = random_rot_v2(data_numpy)
        # if self.bone:
        #     ntu_pairs = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
        #         (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        #         (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
        #         (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12))
        #     bone_data_numpy = np.zeros_like(data_numpy)
        #     for v1, v2 in ntu_pairs:
        #         bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
        #     data_numpy = bone_data_numpy
        # if self.vel:
        #     data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
        #     data_numpy[:, -1] = 0


        return data_numpy, label, index
    

class Feeder_pku(Dataset):
    """ Feeder for dual inputs """

    def __init__(self, data_path, label_path, mmap=True, p_interval=1, simpling='center_crop',
                 window_size=50, random_rot=False):
        self.data_path = data_path
        self.label_path = label_path

        # Random Crop
        self.window_size = window_size
        # Center Crop
        self.simpling = simpling
        self.p_interval = p_interval
        self.random_rot = random_rot
       
        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = self.data[index]

        label = self.label[index]
        valid_frame_num = 50

        # crop
        if self.simpling == "center_crop":
            data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        elif self.simpling == "global_crop":
            data_numpy = tools.crop_resize(data_numpy, valid_frame_num, self.window_size)
        else:
            raise ValueError(f"The value of simpling must be in center_crop or global_crop, but {self.simpling} now")
        if self.random_rot:
            # data_numpy = tools.random_rot(data_numpy)
            data_numpy = random_rot_v2(data_numpy)
        

        return data_numpy, label, index
    

def _rot3d(theta):
    cos, sin = np.cos(theta), np.sin(theta)
    rx = np.array([[1, 0, 0], [0, cos[0], sin[0]], [0, -sin[0], cos[0]]])
    ry = np.array([[cos[1], 0, -sin[1]], [0, 1, 0], [sin[1], 0, cos[1]]])
    rz = np.array([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]])

    rot = np.matmul(rz, np.matmul(ry, rx))
    return rot


def _rot2d(theta):
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[cos, -sin], [sin, cos]])


def random_rot_v2(data, theta=0.3):
    C, T, V, M = data.shape

    if np.all(np.isclose(data, 0)):
        return data
    
    data = data.transpose(3, 1, 2, 0)   # MTVC

    assert C in [2, 3]
    if C == 3:
        theta = np.random.uniform(-theta, theta, size=3)
        rot_mat = _rot3d(theta)
    elif C == 2:
        theta = np.random.uniform(-theta)
        rot_mat = _rot2d(theta)
    data = np.einsum('ab,mtvb->mtva', rot_mat, data)
    data = data.transpose(3, 1, 2, 0)
    return data


ntu60_class_name = [
    "A1. drink water", "A2. eat meal/snack", "A3. brushing teeth", "A4. brushing hair", "A5. drop", "A6. pickup",
    "A7. throw", "A8. sitting down", "A9. standing up (from sitting position)", "A10. clapping", "A11. reading",
    "A12. writing", "A13. tear up paper", "A14. wear jacket", "A15. take off jacket", "A16. wear a shoe",
    "A17. take off a shoe", "A18. wear on glasses", "A19. take off glasses", "A20. put on a hat/cap",
    "A21. take off a hat/cap", "A22. cheer up", "A23. hand waving", "A24. kicking something", "A25. reach into pocket",
    "A26. hopping (one foot jumping)", "A27. jump up", "A28. make a phone call/answer phone", "A29. playing with phone/tablet",
    "A30. typing on a keyboard", "A31. pointing to something with finger", "A32. taking a selfie", "A33. check time (from watch)",
    "A34. rub two hands together", "A35. nod head/bow", "A36. shake head", "A37. wipe face", "A38. salute", "A39. put the palms together",
    "A40. cross hands in front (say stop)", "A41. sneeze/cough", "A42. staggering", "A43. falling", "A44. touch head (headache)",
    "A45. touch chest (stomachache/heart pain)", "A46. touch back (backache)", "A47. touch neck (neckache)", "A48. nausea or vomiting condition",
    "A49. use a fan (with hand or paper)/feeling warm", "A50. punching/slapping other person", "A51. kicking other person",
    "A52. pushing other person", "A53. pat on back of other person", "A54. point finger at the other person",
    "A55. hugging other person", "A56. giving something to other person", "A57. touch other person's pocket",
    "A58. handshaking", "A59. walking towards each other", "A60. walking apart from each other"
]