"""Генерация данных лабораторных работ данного комплекта."""
import numpy as np
from common import ROOT, SEED, save_json

def camera_data(n=15, m=20, seed=SEED):
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 10, (m, 2))
    cameras = rng.uniform(0, 10, (n, 2))
    radius = 3.3
    coverage = np.linalg.norm(points[:,None,:]-cameras[None,:,:],axis=2) <= radius
    # Для каждой точки разрешён хотя бы ближайший кандидат: радиус этой камеры расширяется.
    radii = np.full(n, radius)
    for i in np.flatnonzero(~coverage.any(axis=1)):
        j = np.argmin(np.linalg.norm(cameras-points[i],axis=1))
        radii[j] = max(radii[j], float(np.linalg.norm(cameras[j]-points[i]))+1e-9)
    coverage = np.linalg.norm(points[:,None,:]-cameras[None,:,:],axis=2) <= radii[None,:]
    costs = rng.integers(7, 20, n)
    weights = rng.integers(1, 10, m)
    return {'seed':seed,'points':points.tolist(),'cameras':cameras.tolist(),'radii':radii.tolist(),
            'coverage':coverage.astype(int).tolist(),'costs':costs.tolist(),'weights':weights.tolist(),
            'budget':int(round(0.23*costs.sum()))}


def generate():
    save_json(ROOT/'lab3/data.json', camera_data())
    for n in [10,20,40]: save_json(ROOT/f'lab3/scale_{n}.json',camera_data(n,int(n*4/3),SEED+n))

if __name__=='__main__': generate()
