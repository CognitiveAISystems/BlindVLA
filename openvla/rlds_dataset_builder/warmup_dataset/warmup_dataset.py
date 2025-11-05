from typing import Iterator, Tuple, Any
from pathlib import Path

import glob
import numpy as np
import tensorflow_datasets as tfds


def filter_small_actions(actions, pos_thresh=0.01, rot_thresh=0.06, check_gripper=True):
    actions = np.asarray(actions)
    N = actions.shape[0]
    valid_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        act = actions[i]
        delta_xyz = act[:3]
        delta_euler = act[3:6]
        gripper = act[6]

        pos_movement = np.linalg.norm(delta_xyz)
        rot_movement = np.linalg.norm(delta_euler)

        if pos_thresh is None and rot_thresh is None:
            is_valid = True
        elif pos_thresh is None:
            is_valid = (rot_movement > rot_thresh)
        elif rot_thresh is None:
            is_valid = (pos_movement > pos_thresh)
        else:
            is_valid = (pos_movement > pos_thresh) or (rot_movement > rot_thresh)

        # Preserve gripper toggle events (e.g., from -1 to 1 or vice versa)
        if check_gripper and i > 0 and actions[i - 1][6] != gripper:
            is_valid = True

        valid_mask[i] = is_valid

    return valid_mask

class ExampleDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tasks = [
            {"name": "../../../SimplerEnv/octo_collect/PutCarrotOnPlateInScene-v1/75/data",
             "compressed": False, "filter": False},
            {"name": "../../../ManiSkill/mp_collect/PutOnPlateInScene25Single-v1/75/data",
             "compressed": True, "filter": True},
        ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(480, 640, 3), dtype=np.uint8, encoding_format='jpeg',
                            doc='Observation image.'
                        ),
                    }),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'action': tfds.features.Tensor(shape=(7,), dtype=np.float32, ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(70, spare=5),
            'val': self._generate_examples(5, start=70),
        }

    def _generate_examples(self, num_ep, spare=0, start=0) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path, compressed, use_filter):
            if compressed:
                data = np.load(episode_path, allow_pickle=True)["arr_0"].tolist()
            else:
                data = np.load(episode_path, allow_pickle=True).tolist()

            # prepare data
            ins = data['instruction']
            ins = ins.tolist()[0] if isinstance(ins, np.ndarray) else ins
            actions = data["action"]
            images = np.asarray([np.asarray(img) for img in data["image"]])

            if use_filter:
                mask = filter_small_actions(data["action"])
                actions = actions[mask]
                images = images[mask]
                num_filtered = mask.shape[0] - mask.sum()
                print(f"Filtered {num_filtered}/{mask.shape[0]} actions")
            else:
                num_filtered = 0

            episode = []
            success_count = 0
            for i in range(len(actions)):
                episode.append({
                    'observation': {
                        'image': images[i],
                    },
                    'action': actions[i],
                    'language_instruction': ins,
                })

                if data["info"][i]["success"]:
                    success_count += 1
                else:
                    success_count = 0

                if success_count >= 6:
                    break

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            return sample, num_filtered

        all_files = []
        for task in self.tasks:
            path = Path(task["name"])
            compressed = task["compressed"]
            filt = task["filter"]

            if compressed:
                files = sorted(glob.glob(str(path / "*.npz")))
            else:
                files = sorted(glob.glob(str(path / "*.npy")))

            if spare > 0:
                files = files[:-spare]
            if start > 0:
                start = min(start, len(files) - num_ep)
            files = files[start:start + num_ep]

            print(f"{task}: {len(files)}")

            assert len(files) == num_ep

            all_files.extend([(f, compressed, filt) for f in files])

        num_filtered_total = 0
        for idx, ep_path in enumerate(all_files):
            sample, num_filtered = _parse_example(*ep_path)
            num_filtered_total += num_filtered
            yield ep_path[0], sample

        print(f"Total filtered {num_filtered_total} actions")
