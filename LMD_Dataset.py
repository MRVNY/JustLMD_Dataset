from torch.utils.data import Dataset

from nosmpl.smpl_onnx import SMPLOnnxRuntime
from nosmpl.vis.vis_o3d import Open3DVisualizer

import numpy as np
import torch

#Lyrics_Music_Dance_Dataset
class LMD_Dataset(Dataset): 
    def __init__ (self, dict, indexing):
        self.LMD_Dict = dict
        self.indexing = indexing
        
    def __getitem__(self,index):
        key = self.indexing[str(index)]
        item = self.LMD_Dict[key]
        return item#['lyrics'], item['music'], item['dance']
    
    def __len__ (self):
        return len(self.indexing.keys())
    
    def visualize(self,seq_name):
        seq = self.LMD_Dict[seq_name]
        
        if seq != None:
            poses = seq['dance']
        else:
            print("Sequence does not exist")
            return

        smpl = SMPLOnnxRuntime()
        o3d_vis = Open3DVisualizer(fps=30, enable_axis=False)

        poses = poses.reshape(180,24,3)
        poses = torch.index_select(poses, dim=1, index=torch.arange(0, poses.size(1)-1))
        poses = poses.detach().numpy().astype(np.float32)

        global_orient = [[[0,0,0]]]
        trans = [0,0,0]

        for body in poses:
            data = smpl.forward(body[None], global_orient)

            [vertices, joints, faces] = data
            vertices = vertices[0].squeeze()
            joints = joints[0].squeeze()
            faces = faces.astype(np.int32)

            o3d_vis.update(vertices, faces, trans, R_along_axis=[0, 0, 0], waitKey=1)

        o3d_vis.release()