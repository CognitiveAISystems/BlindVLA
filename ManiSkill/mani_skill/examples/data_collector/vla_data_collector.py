import cv2
import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.controllers.utils.delta_pose import get_numpy, to_numpy
from PIL import Image

class VLADataCollector:
    def __init__(self, env: BaseEnv, camera_name: str, is_image_encode: bool = False, *args, **kwargs,):
        self.env = env.unwrapped
        self.camera_name = camera_name
        self.is_image_encode = is_image_encode

        self.data_dict = self.get_empty_data_dict()


    def get_empty_data_dict(self):
        data_dict = {
            "is_image_encode": self.is_image_encode,
            "image": [],
            "instruction": None,
            "action": [],
            "info": [],
        }
        return data_dict

    def clear_data(self):
        """Clear all collected data."""
        self.data_dict = self.get_empty_data_dict()

    def get_data(self):
        return to_numpy(self.data_dict, self.env.unwrapped.device)

    def save_data(self, save_path, is_compressed=False):
        """Save data as .npy file with dictionary structure."""
        saving_data = to_numpy(self.data_dict, self.env.unwrapped.device)
        saving_data["image"] = [Image.fromarray(im).convert("RGB") for im in saving_data["image"]]
        if is_compressed:
            np.savez_compressed(save_path, saving_data)
            print(f"save data at {save_path}.npz.")
        else:
            np.save(save_path, saving_data)
            print(f"save data at {save_path}.npy.")
        self.clear_data()

    def update_instruction(self):
        if self.data_dict["instruction"] == None:
            self.data_dict["instruction"] = self.env.get_language_instruction()
        else:
            return 
        

    # should run before env.step()
    def update_image(self, camera_name: str=None):
        if camera_name==None:
            rgb = self.env.render().squeeze(0).to(torch.uint8)
        else:
            rgb = self.env.get_obs()['sensor_data'][camera_name]['rgb'].squeeze(0).to(torch.uint8)
        if self.is_image_encode:
            success, encoded_rgb = cv2.imencode('.jpeg', get_numpy(rgb,self.env.unwrapped.device), [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                raise ValueError("JPEG encode error.")
            img_bytes = np.frombuffer(encoded_rgb.tobytes(), dtype=np.uint8)
            rgb = img_bytes

        self.data_dict["image"].append(rgb)

    # should run before env.step()
    def update_action(self, action):
        self.data_dict['action'].append(action)

    def updata_info(self,):
        info = self.env.get_info()
        self.data_dict['info'].append(info)

    # should run before env.step()
    def update_data_dict(self, action):
        self.update_instruction()
        self.update_image(self.camera_name)
        self.updata_info()
        self.update_action(action.astype(np.float32))
