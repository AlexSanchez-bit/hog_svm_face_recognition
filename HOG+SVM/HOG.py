import numpy as np


def normalization(image):
    img = image.astype(np.float64)
    img/=255.0
    return img

def gamma_filter(image,gamma):
    return np.pow(image,gamma)

def compute_gradients(img):
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    return gx,gy
def compute_magnitude_orientation(gx, gy):
    magnitude = np.sqrt(gx**2+gy**2)
    orientation = np.arctan2(gy, gx)
    orientation = np.degrees(orientation) % 180
    return magnitude,orientation


def compute_cells(magnitude,orientation,cell_size,num_bins):
    bin_size = 180/num_bins
    h,w = magnitude.shape

    cells_x=w//cell_size
    cells_y=h//cell_size

    hog_cells = np.zeros((cells_y, cells_x, num_bins), dtype=np.float32)
    for cy in range(cells_y):
        for cx in range(cells_x):
            for iy in range(cell_size):
                for ix in range(cell_size):
                    y = cy * cell_size + iy
                    x = cx * cell_size + ix
                    angle = orientation[y,x]
                    magnitude_ = magnitude[y,x]
                    bin_float = angle/bin_size
                    bin_low = int(bin_float) % num_bins # int por debajo
                    bin_high = int(bin_float+1) % num_bins#int por encima

                    weight_high = bin_float-bin_low
                    weight_low = 1.0 - weight_high

                    hog_cells[cx,cy,bin_low]+=magnitude_ * weight_low
                    hog_cells[cx,cy,bin_high]+=magnitude_ * weight_high
    return hog_cells


def compute_cells_spatial(
    magnitude,
    orientation,
    cell_size=8,
    num_bins=9
):
    h, w = magnitude.shape
    cells_y = h // cell_size
    cells_x = w // cell_size

    hog_cells = np.zeros((cells_y, cells_x, num_bins), dtype=np.float32)
    bin_size = 180.0 / num_bins

    for y in range(h):
        for x in range(w):
            mag = magnitude[y, x]
            if mag == 0:
                continue

            angle = orientation[y, x]

            # --- interpolación angular ---
            bin_f = angle / bin_size
            b0 = int(bin_f) % num_bins
            b1 = (b0 + 1) % num_bins
            wb1 = bin_f - b0
            wb0 = 1.0 - wb1

            # --- coordenadas espaciales ---
            cx_f = (x + 0.5) / cell_size - 0.5
            cy_f = (y + 0.5) / cell_size - 0.5

            cx0 = int(np.floor(cx_f))
            cy0 = int(np.floor(cy_f))

            dx = cx_f - cx0
            dy = cy_f - cy0

            weights = [
                (cx0,     cy0,     (1-dx)*(1-dy)),
                (cx0 + 1, cy0,     dx*(1-dy)),
                (cx0,     cy0 + 1, (1-dx)*dy),
                (cx0 + 1, cy0 + 1, dx*dy)
            ]

            # --- acumular ---
            for cx, cy, ws in weights:
                if 0 <= cx < cells_x and 0 <= cy < cells_y:
                    hog_cells[cy, cx, b0] += mag * ws * wb0
                    hog_cells[cy, cx, b1] += mag * ws * wb1

    return hog_cells

def normalize_blocks(hog_cells, block_size=2, eps=1e-5):
    cells_y, cells_x,_ = hog_cells.shape
    blocks = []

    for y in range(cells_y - block_size + 1):
        for x in range(cells_x - block_size + 1):
            block = hog_cells[y:y+block_size, x:x+block_size]
            v = block.ravel()

            # L2
            norm = np.sqrt(np.sum(v**2) + eps**2)
            v = v / norm

            v = np.minimum(v, 0.2)
            norm = np.sqrt(np.sum(v**2) + eps**2)
            v = v / norm

            blocks.append(v)

    return np.concatenate(blocks)

def hog_descriptor(img,cell_size=8,num_bins=9,block_size=2,gamma=0.5):
    normalized = gamma_filter(normalization(img),gamma)
    gx, gy = compute_gradients(normalized)

    mag, ori = compute_magnitude_orientation(gx, gy)

    hog_cells = compute_cells(
        mag, ori,
        cell_size=cell_size,
        num_bins=num_bins
    )

    hog_vec = normalize_blocks(
     hog_cells,
     block_size=block_size
    )

    return hog_vec
