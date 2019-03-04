import torch.multiprocessing as mp

def test1(x, val):
    x += 1
    val.value += 1
    print(val)

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    num_processes = 16
    val = mp.Value('i', 0)
    x = 0
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=test1, args=(x, val))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()