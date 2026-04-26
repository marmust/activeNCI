import torch

class OnlineKmeans:
    """
    Online k-means clustering over externally-stored latent vectors.

    No local data storage; callers pass latents into retrain() each frame.
    Uses k-means++ seeding for well-separated initial centroids.

    Args:
        num_centroids (int): number of clusters.
        device (torch.device): device for centroid tensors.
        retrain_iters (int): max Lloyd's iterations per retrain call.
        convergence_atol (float): early-stop threshold for centroid shift.
        min_cluster_divisor (int): centroid is considered starved when its point
            count falls below n / (k * divisor). lower = more aggressive reinit.
    """

    def __init__(self, num_centroids, device, retrain_iters=10, convergence_atol=1e-6,
                 min_cluster_divisor=4):
        self.num_centroids = num_centroids
        self.device = device
        self.retrain_iters = retrain_iters
        self.convergence_atol = convergence_atol
        self.min_cluster_divisor = min_cluster_divisor

        self.centroids = None
        self.initialized = False

    def _kmeans_pp_init(self, data):
        """
        Seeds initial centroids using k-means++ (well-separated starts).

        Args:
            data (torch.Tensor): (n, d) latent vectors.

        Returns:
            torch.Tensor: (k, d) initial centroids.
        """
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
        Runs Lloyd's algorithm on the provided latents tensor.

        Starved centroids are reinitialised to the point farthest from any centroid.

        Args:
            latents (torch.Tensor or None): (n, d) latent vectors. no-op if None
                or fewer points than centroids.
        """
        if latents is None or len(latents) < self.num_centroids:
            return

        data = latents.to(self.device)

        # first-time seeding via k-means++
        if not self.initialized:
            self.centroids = self._kmeans_pp_init(data)
            self.initialized = True

        for _ in range(self.retrain_iters):
            dists = torch.cdist(data, self.centroids)
            labels = dists.argmin(dim=1)
            min_cluster_size = max(1, len(data) // (self.num_centroids * self.min_cluster_divisor))

            new_centroids = self.centroids.clone()
            for i in range(self.num_centroids):
                mask = labels == i
                if mask.sum() > min_cluster_size:
                    new_centroids[i] = data[mask].mean(dim=0)
                else:
                    # starved centroid: reinit to the point farthest from all centroids
                    farthest_idx = dists.min(dim=1).values.argmax()
                    new_centroids[i] = data[farthest_idx].clone()

            if torch.allclose(new_centroids, self.centroids, atol=self.convergence_atol):
                self.centroids = new_centroids
                break

            self.centroids = new_centroids

    def classify(self, latent):
        """
        Classifies a single latent vector by nearest centroid.

        Args:
            latent (torch.Tensor): (d,) latent vector.

        Returns:
            tuple: (class_index, dists) or (None, None) if not yet initialized.
        """
        if not self.initialized:
            return None, None

        point = latent.to(self.device).unsqueeze(0)
        dists = torch.cdist(point, self.centroids).squeeze(0)
        class_index = dists.argmin().item()

        return class_index, dists
