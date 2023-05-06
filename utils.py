from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
import gym
import torch
import time
from tqdm import tqdm
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)

def Euccoredist(tar, data):
    v = -torch.norm(tar - data, p='fro', dim=1) 
    vm = v.max()
    return torch.exp(v - vm)
    
def Eucdist(tar, data):
    return -torch.norm(tar - data, p='fro', dim=1)

def Mancoredist(tar, data):
    v = -torch.norm(tar - data, p=1, dim=1)
    vm = v.max()
    return torch.exp(v - vm)

def Mandist(tar, data):
    return -torch.norm(tar - data, p=1, dim=1)

class RegularSaveCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(RegularSaveCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir

    def _init_callback(self) -> None:
        # Create folder if needed
        self.t0 = time.time()
        pass

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print("calls:", self.n_calls, "seconds elapsed:", time.time() - self.t0)
            self.model.save(self.log_dir + "step" + str(self.n_calls))
        return True
        
class RepeatedDataset:
    def __init__(self, datas, batch_size, start_with_random=True):
        self.datas = []
        for data in datas: # list of arrays with the same first dimension.
            self.datas.append(data.clone())
        self.counter, self.idx, self.batch_size = 0, torch.randperm(self.datas[0].shape[0]), batch_size
        if start_with_random:
            for _ in range(len(self.datas)):
                print("shape:", self.datas[_].shape)
                self.datas[_] = self.datas[_][self.idx]
    
    def __len__(self):
        return self.datas[0].shape[0] // self.batch_size    
    
    def getitem(self):
        if self.counter + self.batch_size > len(self.idx):
            self.counter, self.idx = 0, torch.randperm(self.datas[0].shape[0])
            for _ in range(len(self.datas)):
                self.datas[_] = self.datas[_][self.idx]
        ret = []
        for _ in range(len(self.datas)):
            ret.append(self.datas[_][self.counter:self.counter+self.batch_size])
        self.counter += self.batch_size
        """
        print(self.counter, self.counter+self.batch_size)
        
        for _ in range(len(self.datas)):
            print(self.datas[_][self.counter:self.counter+self.batch_size])
        """
        if len(self.datas) == 1: return ret[0]
        else: return ret

if __name__ == "__main__":
    import imageio
    im = imageio.imread("video/walker-walk-da-seed0-actionrepeat4-archresnet18-step0.jpg")
    img = torch.tensor(im).permute(2, 0, 1)
    
    import torchvision.transforms as T
    
    aug1, aug2 = T.RandomAffine(degrees=0, translate=(0.2, 0.2)), T.RandomResizedCrop(size=(84 - 8, 84 - 8))
    img1, img2 = aug1(img).permute(1, 2, 0), aug2(img).permute(1, 2, 0)
    
    imageio.imsave("p1.jpg", img1)
    imageio.imsave("p2.jpg", img2)
    
    
    