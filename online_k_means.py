import torch

class OnlineKmeans:
    """
    Online k-means clustering operating on externally-stored latent vectors.
    No local data storage — reads latents from a LiveDataset via retrain().
    Uses k-means++ initialization for well-separated starting centroids.
    """

    def __init__(self, num_centroids, device, retrain_iters=10, convergence_atol=1e-6):
        self.num_centroids = num_centroids
        self.device = device
        self.retrain_iters = retrain_iters
        self.convergence_atol = convergence_atol

        self.centroids = None
        self.initialized = False

    def _kmeans_pp_init(self, data):
        """K-means++ seeding: pick initial centroids that are well-separated."""
        n = data.size(0)
        k = self.num_centroids

        # first center: random
        centers = [data[torch.randint(n, (1,)).item()]]

        for _ in range(k - 1):
            # distance from each point to its nearest existing center
            C = torch.stack(centers)
            dists = torch.cdist(data, C).min(dim=1).values
            d2 = dists ** 2
            total = d2.sum()

            # sample next center proportional to squared distance
            if total < 1e-10:
                idx = torch.randint(n, (1,)).item()
            else:
                idx = torch.multinomial(d2 / total, 1).item()
            centers.append(data[idx])

        return torch.stack(centers)

    def retrain(self, latents):
        """
        Run Lloyd's algorithm on the provided latents tensor.
        Handles dead centroids by reinitializing them to the farthest point.
        """
        if latents is None or len(latents) < self.num_centroids:
            return

        data = latents.to(self.device)

        # first-time centroid seeding via k-means++
        if not self.initialized:
            self.centroids = self._kmeans_pp_init(data)
            self.initialized = True

        for _ in range(self.retrain_iters):
            dists = torch.cdist(data, self.centroids)
            labels = dists.argmin(dim=1)

            # minimum points per centroid to not be considered starved
            # (each centroid should have at least ~6% of total data)
            min_cluster_size = max(1, len(data) // (self.num_centroids * 4))

            new_centroids = self.centroids.clone()
            for i in range(self.num_centroids):
                mask = labels == i
                if mask.sum() > min_cluster_size:
                    new_centroids[i] = data[mask].mean(dim=0)
                else:
                    # dead or starved centroid: reinitialize to the farthest point
                    min_dists = dists.min(dim=1).values
                    farthest_idx = min_dists.argmax()
                    new_centroids[i] = data[farthest_idx].clone()

            if torch.allclose(new_centroids, self.centroids, atol=self.convergence_atol):
                self.centroids = new_centroids
                break

            self.centroids = new_centroids

    def classify(self, latent):
        """Classify a single latent vector. Returns (class_index, dists) or (None, None)."""
        if not self.initialized:
            return None, None

        point = latent.to(self.device).unsqueeze(0)
        dists = torch.cdist(point, self.centroids).squeeze(0)
        class_index = dists.argmin().item()

        return class_index, dists
