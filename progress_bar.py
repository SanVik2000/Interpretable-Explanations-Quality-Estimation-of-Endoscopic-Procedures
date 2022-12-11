import time
from time import sleep

class ProgressBar(object):
    '''
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='training')
        >>> step = 2
        >>> pbar(step=step)
    '''
    def __init__(self, n_total,width=50,desc = 'Training', epoch=0):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc
        self.epoch = epoch
        self.symbols = ['-' , '|' , '/' , '-' , "\\" , '|' , '/' , '-' , '\\']
        self.cnt = 0

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        if self.epoch != 0:
            bar = f'[\033[93mEpoch {self.epoch}\033[00m : \033[91m{self.desc:>10}\033[00m] {current:>03}/{self.n_total:>03} ['
        else:
            bar = f'[\033[91m{self.desc:>10}\033[00m] {current:>03}/{self.n_total:>03} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            if prog_width == self.width:
                bar += '\033[92m\u2501\033[00m' * (prog_width - 1)
            else:
                bar += '\033[96m\u2501\033[00m' * (prog_width - 1)
            if current< self.n_total:
                symbol = self.symbols[self.cnt % 9]
                bar += f"\033[93m{symbol}\033[00m"
                self.cnt += 1
            else:
                bar += '\033[92m\u2501\033[00m'
        bar += '\u2500' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            time_elapsed = time.time() - self.start_time
            time_elapsed = '%d:%02d' % (time_elapsed // 60, time_elapsed % 60)
            time_info = ' Time: ' + time_elapsed

        if current < self.n_total:
            show_bar += time_info
            if len(info) != 0:
                show_info = f'{show_bar} ' + "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
                print(show_info, end='')
            else:
                print(show_bar, end='')
        else:
            show_bar += time_info
            if len(info) != 0:
                show_info = f'{show_bar} ' + "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
                print(show_info)
            else:
                print(show_bar)


if __name__ == "__main__":
    length=15
    pbar = ProgressBar(n_total=length,desc='Training', epoch=10)
    step = 2
    for i in range(length):
        pbar(step=i, info = {'loss':i, 'accuracy':90})
        sleep(0.25)
    pbar = ProgressBar(n_total=length,desc='Validation', epoch=10)
    step = 2
    for i in range(length):
        pbar(step=i, info = {'loss':i, 'accuracy':90})
        sleep(0.25)
