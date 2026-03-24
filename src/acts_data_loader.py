import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import random
from collections import OrderedDict
#from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
from random import shuffle

random.seed(42)


import warnings
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")

class ActivationsDataset(Dataset):
    """
    A Dataset for loading sharded embedding tensors stored as .pt files.
    Each file contains a tensor of shape [N, 4096] and is loaded as a single batch.
    """
    def __init__(self, acts_dir, ratio_of_training_data=1, file_type='pt', return_input_ids=False, shuffle_data=False):
        """
        Initialize the dataset.
        
        Args:
            acts_dir (str): Directory containing the files
            file_type (str): Type of the files, either 'pt' or 'npy'
            ratio_of_training_data (float, optional): Ratio of files to use. If None, uses all files
        """
        super().__init__()
        self.acts_dir = [acts_dir] if isinstance(acts_dir, str) else acts_dir
        self.file_type = file_type.lower()
        self.ratio_of_training_data = ratio_of_training_data
        self.return_input_ids = return_input_ids
        if self.file_type not in ['pt', 'npy']:
            raise ValueError(f"Unsupported file_type: {self.file_type}")
        
        self.file_paths = []
        for dir_idx, directory in enumerate(self.acts_dir):
            files = sorted([f for f in os.listdir(directory) if f.endswith(f'.{self.file_type}')])
            self.file_paths.extend([(directory, f) for f in files])
        
        print(f"Using {len(self.file_paths)} files")
        # ## Check
        # for fpath, fname in tqdm(self.file_paths):
        #     fpath_input_ids = '/'.join(fpath.split('/')[:-1]) + '/input_ids'
        #     fname_input_ids = fname.replace(f'activations', 'input_ids')
        #     if not os.path.isfile(os.path.join(fpath_input_ids, fname_input_ids)):
        #         print(f"File {fname_input_ids} not found in {fpath_input_ids}")
        
        # we get the input_ids paths
        self.input_ids_paths = []
        for dir_idx, directory in enumerate(self.acts_dir):
            acts_dir_path = Path(directory)
            # change parent to input_ids
            input_ids_dir = str(acts_dir_path.parent / 'input_ids')
            files = sorted([f for f in os.listdir(input_ids_dir) if f.endswith(f'.{self.file_type}')])
            self.input_ids_paths.extend([(input_ids_dir, f) for f in files])

        print('Total number of activations files:', len(self.file_paths))
        print('Total number of input_ids files:', len(self.input_ids_paths))

        # TODO: remove this
        # Remove file containing '1738830147.236417.pt' from both paths
        files_to_remove = ['1740732921.780936.pt', '1739488196.786874.pt']
        self.file_paths = [(d, f) for d, f in self.file_paths if not any(substr in f for substr in files_to_remove)]
        self.input_ids_paths = [(d, f) for d, f in self.input_ids_paths if not any(substr in f for substr in files_to_remove)]

        if self.ratio_of_training_data != 1:
            # we consider a subset (first ratio_of_training_data files)
            self.file_paths = self.file_paths[:int(len(self.file_paths)*self.ratio_of_training_data)]
            self.input_ids_paths = self.input_ids_paths[:int(len(self.input_ids_paths)*self.ratio_of_training_data)]


        print('Total number of activations files:', len(self.file_paths))
        print('Total number of input_ids files:', len(self.input_ids_paths))

        
        #assert len(self.file_paths) == len(self.input_ids_paths), "Number of activation files and input_ids files must match"

        if shuffle_data:
            # Shuffle files in same order as activation files
            # Create a list of paired indices
            indices = list(range(len(self.file_paths)))

            # Shuffle the indices
            shuffle(indices)

            # Reorder both lists using the shuffled indices
            self.file_paths = [self.file_paths[i] for i in indices]
            self.input_ids_paths = [self.input_ids_paths[i] for i in indices]


        # # Analyze file sizes
        # size_counts = defaultdict(int)
        # for directory, filename in tqdm(self.file_paths):
        #     file_path = os.path.join(directory, filename)
        #     try:
        #         if self.file_type == 'pt':
        #             tensor = torch.load(file_path)
        #         else:  # 'npy'
        #             tensor = torch.from_numpy(np.load(file_path))
        #     except Exception as e:
        #         print(f"Error loading file {file_path}: {e}")
        #         continue
        #     size_counts[tensor.shape[0]] += 1
        
        # print(f"\nFound {len(self.file_paths)} files")
        # total_activations = sum(size * count for size, count in size_counts.items())
        # print(f"Total activations: {total_activations}")
        
        if not self.file_paths:
            raise ValueError(f"No {file_type} files found in {acts_dir}")

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle tensors of different sizes.
        Since every file contains a tensor of shape [N, 4096]
        with different N, here we stack them.
        """
        input_ids_in_batch = False
        acts_list = []
        input_ids_list = []
        for element in batch:
            acts_list.append(element[0])
            if element[1] is not None:
                input_ids_in_batch = True
                input_ids_list.append(element[1][0])
        
        acts_batch = torch.cat(acts_list, dim=0)

        if input_ids_in_batch:
            input_ids_batch = torch.cat(input_ids_list, dim=0)
            return acts_batch, input_ids_batch
        else:
            return acts_batch, None
        
    def __len__(self):
        """Return the number of files (each file is a batch)."""
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """
        Get all embeddings from a single file.
        
        Args:
            idx (int): Index of the file to retrieve
            
        Returns:
            torch.Tensor: All embeddings from the file, shape [N, d_in]
            where N varies per file
        """
        if idx >= len(self.file_paths):
            raise IndexError(f"Index {idx} out of range")
        
        directory, filename = self.file_paths[idx]
        file_path = os.path.join(directory, filename)

        if self.file_type == 'pt':
            tensor = torch.load(file_path)
        else:  # 'npy'
            tensor = torch.from_numpy(np.load(file_path))
        
        if self.return_input_ids:
            input_ids_dir, input_ids_filename = self.input_ids_paths[idx]
            input_ids_file_path = os.path.join(input_ids_dir, input_ids_filename)
            if self.file_type == 'pt':
                input_ids_tensor = torch.load(input_ids_file_path)
            else:  # 'npy'
                input_ids_tensor = torch.from_numpy(np.load(input_ids_file_path))
            return tensor, input_ids_tensor
            
        return (tensor, None)