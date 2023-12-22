import numpy as np


class Sampler:
    """
    PyTorch의 DataLoader처럼 반복마다 batch_size만큼의 미니 배치를 반환하는 클래스이다.
    shuffle이 False면 순서대로 batch_size만큼, True면 랜덤으로 섞은 batch_size의 미니 배치를 반환한다.
    """
    def __init__(self, data: np.ndarray, label: np.ndarray, batch_size=1, shuffle=False):
        """
        shuffle이 True면 데이터셋 길이만큼의 랜덤 인덱스 배열을 만든다.
        :param data: 데이터 배열
        :param label: 레이블 배열
        :param batch_size: 배치 크기
        :param shuffle: 랜덤 여부
        """
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.index = np.arange(label.shape[0])
        if shuffle:
            np.random.shuffle(self.index)

        self.current_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        # 데이터셋의 끝까지 순회하지 않았으면
        if self.current_idx < self.label.shape[0]:
            start = self.current_idx
            self.current_idx += self.batch_size
            # 시작 인덱스부터 ((시작 인덱스 + batch_size), (끝 인덱스 - 1) 중 작은 값)까지 데이터를 반환한다.
            index = self.index[start:min(self.current_idx, self.label.shape[0])]
            return self.data[index], self.label[index]
        # 데이터셋 끝까지 순회했으면
        else:
            # 다시 처음부터
            self.current_idx = 0
            # 랜덤이면 다시 섞기
            if self.shuffle:
                np.random.shuffle(self.index)

            raise StopIteration
