# Bin non-zero values and remove outliers
counts, bins = np.histogram(depth)
left_bin, right_bin = bins[np.argmax(counts)], bins[np.argmax(counts) + 1]
idcs = np.where(np.logical_and(depth >= left_bin, depth <= right_bin))
detector_plane_pts = coords[idcs]

median = np.median(detector_plane_pts[:, 2])

# Apply PCA and extract plane
pca = sklearn.decomposition.PCA()
pca.fit(detector_plane_pts)

self.detector_plane["normal_vector"] = np.round(pca.components_[-1], 1)
self.detector_plane["centroid"] = np.round(detector_plane_pts.mean(axis=0), 2)

a, b, c = self.detector_plane["normal_vector"]
x0, y0, z0 = self.detector_plane["centroid"]
plane_eq = f"{a:.2f}(x - {x0:.2f}) + {b:.2f}(y - {y0:.2f}) + {c:.2f}(z - {z0:.2f}) = 0"

print(f"Detector plane is set: {plane_eq}")

self.viz_matplotlib(detector_plane_pts)