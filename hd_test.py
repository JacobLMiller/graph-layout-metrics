import numpy as np 
# from sklearn.decomposition import PCA
# from sklearn.datasets import load_iris
import anndata as ad
import scanpy as sc # how to read the data
from concurrent.futures import ProcessPoolExecutor
import pickle

from modules.metrics import MetricsData

def filter_batches(adata):
    adatas = []
    for c in adata.obs['tech'].cat.categories:
        new_adata = ad.AnnData(
            X=adata.layers['counts'][adata.obs['tech'] == c, :],
            obs=adata.obs[adata.obs['tech'] == c]
        )
        new_adata.layers['counts'] = new_adata.X
        adatas.append(new_adata)
    return adatas

if __name__ == "__main__":
    # get data
        # 

    # import h5py
    # f = h5py.File("human_pancreas_norm_complexBatch.h5ad")

    adata = sc.read_h5ad("human_pancreas_norm_complexBatch.h5ad")
    adatas = filter_batches(adata)

    # print(adatas[0])
    # parallel proecssing with Kruskal's stress
    futures = {}
    with ProcessPoolExecutor() as executor: # do multiple processes
        for i in range(len(adatas)):
            for j in range(i+1,len(adatas)):
                # the key to the futures dictionary is (batch1_name, batch2_name)
                print(adatas[i].layers['counts'].shape)
                futures[(adatas[i].obs['tech'].iloc[0], adatas[j].obs['tech'].iloc[0])] = executor.submit(
                    lambda M: M.compute_stress_kruskal(), 
                    MetricsData(
                        np.transpose(adatas[i].layers['counts']), # it doesn't matter which is considered the HD data
                        np.transpose(adatas[j].layers['counts'])
                    )
                )
        executor.shutdown(wait=True)

    # get Kruskal's stress scores
    futures = {k: v.result() for k, v in futures.items()}

    # store dictionary
    with open("pancreas-batch-stress-scores.pkl", "wb") as f:
        pickle.dump(futures, f)